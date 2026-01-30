"""Trainer for the ResNet classifier."""

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
)

from ..configs import ClassifierConfig
from .base import BaseTrainer


class ClassifierTrainer(BaseTrainer):
    """Trainer for classification models."""

    def __init__(
        self,
        model: nn.Module,
        config: ClassifierConfig,
        class_weights: Optional[torch.Tensor] = None,
        idx_to_class: Optional[dict[int, str]] = None,
    ):
        """Initialize classifier trainer.

        Args:
            model: Classification model.
            config: Classifier configuration.
            class_weights: Optional class weights for loss function.
            idx_to_class: Mapping from class indices to names.
        """
        super().__init__(model, config, idx_to_class)

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

        # Additional tracking
        self.best_accuracy = 0.0

    def train_epoch(self, train_loader: DataLoader) -> dict:
        """Train for one epoch."""
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
                    f"Loss={loss.item():.4f}, Acc={100.0 * correct / total:.2f}%"
                )

        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
        }

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, return_predictions: bool = False) -> dict:
        """Evaluate the model."""
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

    def _add_checkpoint_extras(self, checkpoint: dict):
        """Add classifier-specific items to checkpoint."""
        checkpoint["best_accuracy"] = self.best_accuracy

    def _load_checkpoint_extras(self, checkpoint: dict):
        """Load classifier-specific items from checkpoint."""
        self.best_accuracy = checkpoint.get("best_accuracy", 0.0)

    def print_classification_report(self, data_loader: DataLoader):
        """Print detailed classification report."""
        metrics = self.evaluate(data_loader, return_predictions=True)

        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)

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

        print("\nOverall Metrics:")
        print(f"  Accuracy: {100*metrics['accuracy']:.2f}%")
        print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"  Recall (macro): {metrics['recall_macro']:.4f}")

        return metrics
