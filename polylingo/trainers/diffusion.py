"""Trainer for the Diffusion Model."""

import time
from typing import Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from ..configs import DiffusionConfig
from ..models import GaussianDiffusion
from ..utils import EMA, get_cosine_schedule_with_warmup
from .base import BaseTrainer


class DiffusionTrainer(BaseTrainer):
    """Trainer for Diffusion Model."""

    def __init__(
        self,
        diffusion: GaussianDiffusion,
        config: DiffusionConfig,
        idx_to_class: Optional[dict[int, str]] = None,
    ):
        """Initialize diffusion trainer.

        Args:
            diffusion: GaussianDiffusion model.
            config: Diffusion configuration.
            idx_to_class: Mapping from class indices to names.
        """
        # Store diffusion and extract the underlying model for base class
        self.diffusion = diffusion
        super().__init__(diffusion.model, config, idx_to_class)

        # Move full diffusion module to device
        self.diffusion = self.diffusion.to(self.device)

        # Optimizer (on the denoising model)
        self.optimizer = AdamW(
            diffusion.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # EMA
        self.ema = EMA(diffusion.model, decay=config.ema_decay)

        # Tracking
        self.global_step = 0

        # Create samples directory
        self.samples_dir = config.checkpoint_dir / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, train_loader: DataLoader, scheduler=None) -> dict:
        """Train for one epoch."""
        self.diffusion.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Classifier-free guidance: randomly drop labels
            if self.config.use_cfg and self.config.cfg_dropout > 0:
                drop_mask = torch.rand(labels.size(0), device=self.device) < self.config.cfg_dropout
                labels = torch.where(
                    drop_mask,
                    torch.full_like(labels, self.diffusion.model.null_class),
                    labels,
                )

            # Compute loss
            self.optimizer.zero_grad()
            loss = self.diffusion.training_loss(images, labels)

            # Backward
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.diffusion.model.parameters(),
                    self.config.gradient_clip,
                )

            self.optimizer.step()

            # Update EMA
            self.ema.update(self.diffusion.model)

            # Update scheduler
            if scheduler is not None:
                scheduler.step()

            # Track
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg": f"{total_loss / num_batches:.4f}",
            })

        return {"loss": total_loss / num_batches}

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> dict:
        """Evaluate the model (compute loss on data)."""
        self.diffusion.model.eval()
        total_loss = 0.0
        num_batches = 0

        for images, labels in data_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            loss = self.diffusion.training_loss(images, labels)
            total_loss += loss.item()
            num_batches += 1

        return {"loss": total_loss / num_batches}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> dict:
        """Full training loop."""
        num_epochs = self.config.num_epochs

        # Create scheduler
        num_training_steps = len(train_loader) * num_epochs
        scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps,
        )

        print(f"\nStarting diffusion training for {num_epochs} epochs...")
        print(f"Total training steps: {num_training_steps}")
        print(f"Warmup steps: {self.config.warmup_steps}")

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, scheduler)
            self._update_history("train", train_metrics)

            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]
            train_loss = train_metrics["loss"]

            print(
                f"Epoch {epoch}/{num_epochs}: "
                f"Loss={train_loss:.4f}, LR={current_lr:.6f}, Time={epoch_time:.1f}s"
            )

            # Save best model
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self.save_checkpoint("best_diffusion.pt")
                print("  -> Saved new best model!")

            # Generate samples periodically
            if epoch % self.config.sample_interval == 0:
                self.generate_samples(epoch, use_ddim=True, ddim_steps=50)
                print("  -> Generated samples")

            # Save checkpoint periodically
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(f"diffusion_epoch_{epoch:03d}.pt")

        print(f"\nTraining complete! Best loss: {self.best_loss:.4f}")
        return self.history

    @torch.no_grad()
    def generate_samples(
        self,
        epoch: int,
        num_samples: int = 64,
        use_ddim: bool = False,
        ddim_steps: int = 50,
    ):
        """Generate and save samples."""
        self.diffusion.model.eval()

        # Apply EMA weights
        self.ema.apply(self.diffusion.model)

        # Generate unconditional samples
        print("  Generating unconditional samples...")
        if use_ddim:
            samples = self.diffusion.sample_ddim(
                num_samples, None, cfg_scale=1.0, ddim_steps=ddim_steps,
                device=self.device, show_progress=False,
            )
        else:
            samples = self.diffusion.sample(
                num_samples, None, cfg_scale=1.0,
                device=self.device, show_progress=False,
            )

        # Rescale from [-1, 1] to [0, 1]
        samples = (samples + 1) / 2

        save_image(
            samples,
            self.samples_dir / f"unconditional_epoch_{epoch:03d}.png",
            nrow=8,
            normalize=False,
        )

        # Generate class-conditional samples
        if self.config.conditional:
            print("  Generating class-conditional samples...")
            num_classes_to_show = min(64, self.config.num_classes)
            class_labels = torch.arange(num_classes_to_show, device=self.device)

            if use_ddim:
                cond_samples = self.diffusion.sample_ddim(
                    num_classes_to_show, class_labels, cfg_scale=self.config.cfg_scale,
                    ddim_steps=ddim_steps, device=self.device, show_progress=False,
                )
            else:
                cond_samples = self.diffusion.sample(
                    num_classes_to_show, class_labels, cfg_scale=self.config.cfg_scale,
                    device=self.device, show_progress=False,
                )

            cond_samples = (cond_samples + 1) / 2

            save_image(
                cond_samples,
                self.samples_dir / f"conditional_epoch_{epoch:03d}.png",
                nrow=8,
                normalize=False,
            )

        # Restore original weights
        self.ema.restore(self.diffusion.model)

    def _add_checkpoint_extras(self, checkpoint: dict):
        """Add diffusion-specific items to checkpoint."""
        checkpoint["ema_state"] = self.ema.state_dict()
        checkpoint["global_step"] = self.global_step

    def _load_checkpoint_extras(self, checkpoint: dict):
        """Load diffusion-specific items from checkpoint."""
        if "ema_state" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state"])
        elif "ema_shadow" in checkpoint:
            # Legacy format
            self.ema.shadow = checkpoint["ema_shadow"]
        self.global_step = checkpoint.get("global_step", 0)

    def save_checkpoint(self, filename: str):
        """Save checkpoint with EMA weights applied."""
        # Apply EMA for saving
        self.ema.apply(self.diffusion.model)
        super().save_checkpoint(filename)
        # Restore original weights
        self.ema.restore(self.diffusion.model)
