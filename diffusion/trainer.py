"""Training utilities for the diffusion model."""

import copy
import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from .config import DiffusionConfig
from .diffusion import GaussianDiffusion


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


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        """Update shadow parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply(self, model: nn.Module):
        """Apply shadow parameters to model."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model: nn.Module):
        """Restore original parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
):
    """Create a schedule with warmup and cosine decay."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class DiffusionTrainer:
    """Trainer for Diffusion Model."""

    def __init__(
        self,
        diffusion: GaussianDiffusion,
        config: DiffusionConfig,
        idx_to_class: dict[int, str],
    ):
        self.diffusion = diffusion
        self.config = config
        self.idx_to_class = idx_to_class

        # Device
        self.device = get_device(config.device)
        print(f"Using device: {self.device}")
        self.diffusion = self.diffusion.to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            diffusion.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # EMA
        self.ema = EMA(diffusion.model, decay=config.ema_decay)

        # Tracking
        self.global_step = 0
        self.best_loss = float("inf")
        self.history = {
            "train_loss": [],
            "lr": [],
        }

        # Create samples directory
        self.samples_dir = config.checkpoint_dir / "samples"
        self.samples_dir.mkdir(exist_ok=True)

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        scheduler=None,
    ) -> float:
        """Train for one epoch."""
        self.diffusion.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Classifier-free guidance: randomly drop labels
            if self.config.use_cfg and self.config.cfg_dropout > 0:
                drop_mask = torch.rand(labels.size(0), device=self.device) < self.config.cfg_dropout
                # Set dropped labels to null class
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

            # Log periodically
            if (batch_idx + 1) % self.config.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.history["lr"].append(current_lr)

        return total_loss / num_batches

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

        # Generate class-conditional samples (one per class, up to 64)
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

    def train(
        self,
        train_loader: DataLoader,
        num_epochs: Optional[int] = None,
    ) -> dict:
        """Full training loop."""
        if num_epochs is None:
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
            train_loss = self.train_epoch(train_loader, epoch, scheduler)
            self.history["train_loss"].append(train_loss)

            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch}/{num_epochs}: "
                f"Loss={train_loss:.4f}, LR={current_lr:.6f}, Time={epoch_time:.1f}s"
            )

            # Save best model
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self.save_checkpoint("best_diffusion.pt")
                print(f"  -> Saved new best model!")

            # Generate samples periodically
            if epoch % self.config.sample_interval == 0:
                self.generate_samples(epoch, use_ddim=True, ddim_steps=50)
                print(f"  -> Generated samples")

            # Save checkpoint periodically
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(f"diffusion_epoch_{epoch:03d}.pt")

        print(f"\nTraining complete! Best loss: {self.best_loss:.4f}")
        return self.history

    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        # Apply EMA for saving
        self.ema.apply(self.diffusion.model)

        checkpoint_path = self.config.checkpoint_dir / filename
        torch.save({
            "model_state_dict": self.diffusion.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "ema_shadow": self.ema.shadow,
            "config": self.config,
            "idx_to_class": self.idx_to_class,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "history": self.history,
        }, checkpoint_path)

        # Restore original weights
        self.ema.restore(self.diffusion.model)

    def load_checkpoint(self, filename: str):
        """Load checkpoint."""
        checkpoint_path = self.config.checkpoint_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.diffusion.model.load_state_dict(checkpoint["model_state_dict"])
        self.ema.shadow = checkpoint.get("ema_shadow", self.ema.shadow)
        self.idx_to_class = checkpoint.get("idx_to_class", {})
        self.global_step = checkpoint.get("global_step", 0)
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        self.history = checkpoint.get("history", self.history)
