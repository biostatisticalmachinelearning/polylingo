"""Training and evaluation utilities."""

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

from .config import Config
from .model import EarlyStopping, get_device


class Trainer:
    """Trainer class for model training and evaluation."""

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        class_weights: Optional[torch.Tensor] = None,
        idx_to_class: Optional[dict[int, str]] = None,
    ):
        """
        Args:
            model: PyTorch model to train.
            config: Training configuration.
            class_weights: Optional class weights for loss function.
            idx_to_class: Mapping from class indices to names.
        """
        self.model = model
        self.config = config
        self.idx_to_class = idx_to_class or {}

        # Device
        self.device = get_device(config.device)
        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)

        # Loss function with optional class weights
        if config.use_weighted_loss and class_weights is not None:
            class_weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            print("Using weighted cross-entropy loss")
        else:
            self.criterion = nn.CrossEntropyLoss()
            print("Using standard cross-entropy loss")

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
        else:
            self.scheduler = StepLR(
                self.optimizer,
                step_size=config.lr_step_size,
                gamma=config.lr_gamma,
            )

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
        )

        # Tracking
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1_macro": [],
            "lr": [],
        }

    def train_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader.

        Returns:
            Tuple of (average loss, accuracy).
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Log progress
            if (batch_idx + 1) % self.config.log_interval == 0:
                print(
                    f"  Batch {batch_idx + 1}/{len(train_loader)}: "
                    f"Loss={loss.item():.4f}, "
                    f"Acc={100.0 * correct / total:.2f}%"
                )

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader,
        return_predictions: bool = False,
    ) -> dict:
        """Evaluate the model.

        Args:
            data_loader: Data loader for evaluation.
            return_predictions: Whether to return predictions and labels.

        Returns:
            Dictionary of evaluation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        for images, labels in data_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        # Compute metrics
        metrics = {
            "loss": total_loss / len(all_labels),
            "accuracy": accuracy_score(all_labels, all_predictions),
            "f1_macro": f1_score(all_labels, all_predictions, average="macro", zero_division=0),
            "f1_weighted": f1_score(all_labels, all_predictions, average="weighted", zero_division=0),
            "precision_macro": precision_score(all_labels, all_predictions, average="macro", zero_division=0),
            "recall_macro": recall_score(all_labels, all_predictions, average="macro", zero_division=0),
        }

        if return_predictions:
            metrics["predictions"] = all_predictions
            metrics["labels"] = all_labels

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict:
        """Full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.

        Returns:
            Training history dictionary.
        """
        print(f"\nStarting training for {self.config.num_epochs} epochs...")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs} (LR: {current_lr:.6f})")
            print("-" * 50)

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Evaluate
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]
            val_f1 = val_metrics["f1_macro"]

            # Update learning rate
            self.scheduler.step()

            # Track history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1_macro"].append(val_f1)
            self.history["lr"].append(current_lr)

            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(
                f"  Train Loss: {train_loss:.4f}, Train Acc: {100*train_acc:.2f}%\n"
                f"  Val Loss: {val_loss:.4f}, Val Acc: {100*val_acc:.2f}%, "
                f"Val F1 (macro): {val_f1:.4f}\n"
                f"  Time: {epoch_time:.1f}s"
            )

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.save_checkpoint("best_model.pt")
                print("  -> Saved new best model!")

            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        print(f"\nTraining complete! Best Val Acc: {100*self.best_val_acc:.2f}%")
        return self.history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint.

        Args:
            filename: Checkpoint filename.
        """
        checkpoint_path = self.config.checkpoint_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "idx_to_class": self.idx_to_class,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "history": self.history,
        }, checkpoint_path)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint.

        Args:
            filename: Checkpoint filename.
        """
        checkpoint_path = self.config.checkpoint_dir / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.idx_to_class = checkpoint.get("idx_to_class", {})
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        self.history = checkpoint.get("history", self.history)

    def print_classification_report(self, data_loader: DataLoader):
        """Print detailed classification report.

        Args:
            data_loader: Data loader for evaluation.
        """
        metrics = self.evaluate(data_loader, return_predictions=True)

        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)

        # Get class names
        target_names = [
            self.idx_to_class.get(i, str(i))
            for i in range(self.config.num_classes)
        ]

        print(classification_report(
            metrics["labels"],
            metrics["predictions"],
            target_names=target_names,
            zero_division=0,
        ))

        # Overall metrics
        print("\nOverall Metrics:")
        print(f"  Accuracy: {100*metrics['accuracy']:.2f}%")
        print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"  Recall (macro): {metrics['recall_macro']:.4f}")

        return metrics
