"""Training utilities for the VAE."""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from .config import VAEConfig
from .model import VAE, vae_loss


def get_device(device_str: str = "auto") -> torch.device:
    """Get the best available device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class VAETrainer:
    """Trainer for Variational Autoencoder."""

    def __init__(
        self,
        model: VAE,
        config: VAEConfig,
        idx_to_class: dict[int, str] = None,
    ):
        self.model = model
        self.config = config
        self.idx_to_class = idx_to_class or {}

        # Device
        self.device = get_device(config.device)
        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)

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
                mode='min',
                factor=0.5,
                patience=config.lr_patience,
            )
        else:
            self.scheduler = None

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
        )

        # Tracking
        self.best_val_loss = float("inf")
        self.history = {
            "train_loss": [],
            "train_recon": [],
            "train_kl": [],
            "val_loss": [],
            "val_recon": [],
            "val_kl": [],
            "lr": [],
        }

        # Create samples directory
        self.samples_dir = config.checkpoint_dir / "samples"
        self.samples_dir.mkdir(exist_ok=True)

        # Fixed noise for consistent sample generation
        self.fixed_noise = torch.randn(64, config.latent_dim, device=self.device)

    def train_epoch(self, train_loader: DataLoader) -> tuple[float, float, float]:
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
                images = batch
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

        return (
            total_loss / num_batches,
            total_recon / num_batches,
            total_kl / num_batches,
        )

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> tuple[float, float, float]:
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
                images = batch
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

        return (
            total_loss / num_batches,
            total_recon / num_batches,
            total_kl / num_batches,
        )

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
            images = batch
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

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict:
        """Full training loop."""
        print(f"\nStarting VAE training for {self.config.num_epochs} epochs...")
        print(f"Latent dimension: {self.config.latent_dim}, Beta: {self.config.beta}")

        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs} (LR: {current_lr:.6f})")
            print("-" * 50)

            # Train
            train_loss, train_recon, train_kl = self.train_epoch(train_loader)

            # Evaluate
            val_loss, val_recon, val_kl = self.evaluate(val_loader)

            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Track history
            self.history["train_loss"].append(train_loss)
            self.history["train_recon"].append(train_recon)
            self.history["train_kl"].append(train_kl)
            self.history["val_loss"].append(val_loss)
            self.history["val_recon"].append(val_recon)
            self.history["val_kl"].append(val_kl)
            self.history["lr"].append(current_lr)

            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(
                f"  Train: Loss={train_loss:.4f}, Recon={train_recon:.4f}, KL={train_kl:.4f}\n"
                f"  Val:   Loss={val_loss:.4f}, Recon={val_recon:.4f}, KL={val_kl:.4f}\n"
                f"  Time: {epoch_time:.1f}s"
            )

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_vae.pt")
                print("  -> Saved new best model!")

            # Generate samples periodically
            if (epoch + 1) % self.config.sample_interval == 0:
                self.generate_samples(epoch + 1, val_loader)
                print(f"  -> Generated samples")

            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        print(f"\nTraining complete! Best Val Loss: {self.best_val_loss:.4f}")
        return self.history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_path = self.config.checkpoint_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "idx_to_class": self.idx_to_class,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }, checkpoint_path)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = self.config.checkpoint_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.idx_to_class = checkpoint.get("idx_to_class", {})
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.history = checkpoint.get("history", self.history)
