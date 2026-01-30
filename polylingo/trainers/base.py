"""Base trainer class with common functionality."""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils import get_device, EarlyStopping


class BaseTrainer(ABC):
    """Abstract base trainer with common training utilities."""

    def __init__(
        self,
        model: nn.Module,
        config: Any,
        idx_to_class: Optional[dict[int, str]] = None,
    ):
        """Initialize trainer.

        Args:
            model: PyTorch model to train.
            config: Training configuration.
            idx_to_class: Mapping from class indices to names.
        """
        self.model = model
        self.config = config
        self.idx_to_class = idx_to_class or {}

        # Device
        self.device = get_device(config.device)
        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)

        # Initialize optimizer (to be set by subclass)
        self.optimizer = None
        self.scheduler = None

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
        )

        # Tracking
        self.best_loss = float("inf")
        self.history = {}

    @abstractmethod
    def train_epoch(self, train_loader: DataLoader) -> dict:
        """Train for one epoch.

        Args:
            train_loader: Training data loader.

        Returns:
            Dictionary of training metrics.
        """
        pass

    @abstractmethod
    def evaluate(self, data_loader: DataLoader) -> dict:
        """Evaluate the model.

        Args:
            data_loader: Data loader for evaluation.

        Returns:
            Dictionary of evaluation metrics.
        """
        pass

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> dict:
        """Full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Optional validation data loader.

        Returns:
            Training history dictionary.
        """
        print(f"\nStarting training for {self.config.num_epochs} epochs...")

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
                val_loss = val_metrics.get("loss", float("inf"))
            else:
                val_loss = train_metrics.get("loss", float("inf"))

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Print epoch summary
            epoch_time = time.time() - epoch_start
            self._print_epoch_summary(epoch, train_metrics, val_metrics if val_loader else None, epoch_time)

            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint("best_model.pt")
                print("  -> Saved new best model!")

            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\nEarly stopping at epoch {epoch}")
                break

        print(f"\nTraining complete! Best loss: {self.best_loss:.4f}")
        return self.history

    def _update_history(self, prefix: str, metrics: dict):
        """Update history with metrics."""
        for key, value in metrics.items():
            history_key = f"{prefix}_{key}"
            if history_key not in self.history:
                self.history[history_key] = []
            self.history[history_key].append(value)

    def _print_epoch_summary(
        self,
        epoch: int,
        train_metrics: dict,
        val_metrics: Optional[dict],
        epoch_time: float,
    ):
        """Print epoch summary."""
        train_str = ", ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
        print(f"  Train: {train_str}")
        if val_metrics:
            val_str = ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            print(f"  Val:   {val_str}")
        print(f"  Time: {epoch_time:.1f}s")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint.

        Args:
            filename: Checkpoint filename.
        """
        checkpoint_path = self.config.checkpoint_dir / filename
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "idx_to_class": self.idx_to_class,
            "best_loss": self.best_loss,
            "history": self.history,
        }
        self._add_checkpoint_extras(checkpoint)
        torch.save(checkpoint, checkpoint_path)

    def _add_checkpoint_extras(self, checkpoint: dict):
        """Add extra items to checkpoint. Override in subclass."""
        pass

    def load_checkpoint(self, filename: str):
        """Load model checkpoint.

        Args:
            filename: Checkpoint filename.
        """
        checkpoint_path = self.config.checkpoint_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.idx_to_class = checkpoint.get("idx_to_class", {})
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        self.history = checkpoint.get("history", {})
        self._load_checkpoint_extras(checkpoint)

    def _load_checkpoint_extras(self, checkpoint: dict):
        """Load extra items from checkpoint. Override in subclass."""
        pass
