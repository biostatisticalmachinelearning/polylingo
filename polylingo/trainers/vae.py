"""Trainer for the Variational Autoencoder."""

import time
from typing import Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from ..configs import VAEConfig
from ..models import ConvVAE, ConditionalConvVAE, vae_loss
from .base import BaseTrainer


class VAETrainer(BaseTrainer):
    """Trainer for Variational Autoencoder."""

    def __init__(
        self,
        model: ConvVAE,
        config: VAEConfig,
        idx_to_class: Optional[dict[int, str]] = None,
    ):
        """Initialize VAE trainer.

        Args:
            model: VAE model.
            config: VAE configuration.
            idx_to_class: Mapping from class indices to names.
        """
        super().__init__(model, config, idx_to_class)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        if config.lr_scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.num_epochs,
                eta_min=config.learning_rate * 0.01,
            )
        elif config.lr_scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=config.lr_patience,
            )
        else:
            self.scheduler = None

        # Create samples directory
        self.samples_dir = config.checkpoint_dir / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        # Fixed noise for consistent sample generation
        self.fixed_noise = torch.randn(64, config.latent_dim, device=self.device)

    def train_epoch(self, train_loader: DataLoader) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            if self.config.conditional:
                images, labels = batch
                labels = labels.to(self.device)
            else:
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                labels = None

            images = images.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if self.config.conditional:
                reconstruction, mu, log_var = self.model(images, labels)
            else:
                reconstruction, mu, log_var = self.model(images)

            # Compute loss
            loss, recon_loss, kl_loss = vae_loss(
                reconstruction, images, mu, log_var, self.config.beta
            )

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            num_batches += 1

            # Log progress
            if (batch_idx + 1) % self.config.log_interval == 0:
                print(
                    f"  Batch {batch_idx + 1}/{len(train_loader)}: "
                    f"Loss={loss.item():.4f}, Recon={recon_loss.item():.4f}, "
                    f"KL={kl_loss.item():.4f}"
                )

        return {
            "loss": total_loss / num_batches,
            "recon": total_recon / num_batches,
            "kl": total_kl / num_batches,
        }

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> dict:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_batches = 0

        for batch in data_loader:
            if self.config.conditional:
                images, labels = batch
                labels = labels.to(self.device)
            else:
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                labels = None

            images = images.to(self.device)

            if self.config.conditional:
                reconstruction, mu, log_var = self.model(images, labels)
            else:
                reconstruction, mu, log_var = self.model(images)

            loss, recon_loss, kl_loss = vae_loss(
                reconstruction, images, mu, log_var, self.config.beta
            )

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "recon": total_recon / num_batches,
            "kl": total_kl / num_batches,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> dict:
        """Full training loop with sample generation."""
        print(f"\nStarting VAE training for {self.config.num_epochs} epochs...")
        print(f"Latent dimension: {self.config.latent_dim}, Beta: {self.config.beta}")

        for epoch in range(1, self.config.num_epochs + 1):
            epoch_start = time.time()
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(f"\nEpoch {epoch}/{self.config.num_epochs} (LR: {current_lr:.6f})")
            print("-" * 50)

            # Train
            train_metrics = self.train_epoch(train_loader)
            self._update_history("train", train_metrics)

            # Evaluate
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                self._update_history("val", val_metrics)
                val_loss = val_metrics["loss"]
            else:
                val_loss = train_metrics["loss"]

            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Print epoch summary
            epoch_time = time.time() - epoch_start
            self._print_epoch_summary(epoch, train_metrics, val_metrics if val_loader else None, epoch_time)

            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint("best_vae.pt")
                print("  -> Saved new best model!")

            # Generate samples periodically
            if epoch % self.config.sample_interval == 0:
                self.generate_samples(epoch, val_loader or train_loader)
                print("  -> Generated samples")

            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\nEarly stopping at epoch {epoch}")
                break

        print(f"\nTraining complete! Best loss: {self.best_loss:.4f}")
        return self.history

    @torch.no_grad()
    def generate_samples(self, epoch: int, data_loader: DataLoader):
        """Generate and save sample reconstructions and random samples."""
        self.model.eval()

        # Get a batch for reconstruction
        batch = next(iter(data_loader))
        if self.config.conditional:
            images, labels = batch
            labels = labels.to(self.device)
        else:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            labels = None

        images = images[:32].to(self.device)

        # Reconstruct
        if self.config.conditional:
            reconstructions, _, _ = self.model(images, labels[:32] if labels is not None else None)
        else:
            reconstructions, _, _ = self.model(images)

        # Interleave originals and reconstructions
        comparison = torch.cat([images[:16], reconstructions[:16]])
        save_image(
            comparison,
            self.samples_dir / f"reconstruction_epoch_{epoch:03d}.png",
            nrow=16,
            normalize=False,
        )

        # Generate random samples
        samples = self.model.sample(64, self.device)
        save_image(
            samples,
            self.samples_dir / f"samples_epoch_{epoch:03d}.png",
            nrow=8,
            normalize=False,
        )

        # Generate with fixed noise (to see progression)
        fixed_samples = self.model.decode(self.fixed_noise)
        save_image(
            fixed_samples,
            self.samples_dir / f"fixed_samples_epoch_{epoch:03d}.png",
            nrow=8,
            normalize=False,
        )
