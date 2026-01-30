"""Model definitions for Unicode character classification."""

import torch
import torch.nn as nn
from torchvision import models


def create_model(
    num_classes: int,
    model_name: str = "resnet18",
    pretrained: bool = True,
) -> nn.Module:
    """Create a ResNet model for character classification.

    Args:
        num_classes: Number of output classes.
        model_name: Name of the model architecture.
        pretrained: Whether to use pretrained weights.

    Returns:
        PyTorch model.
    """
    # Select model architecture
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
    elif model_name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        in_features = model.fc.in_features
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Modify first conv layer to handle 64x64 input better
    # Original ResNet expects 224x224, but we can use it with 64x64
    # The model will still work, just with smaller feature maps

    # Replace classifier head
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )

    return model


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        """
        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop.

        Args:
            val_loss: Current validation loss.

        Returns:
            True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def get_device(device_str: str = "auto") -> torch.device:
    """Get the best available device.

    Args:
        device_str: Device specification ("auto", "cuda", "mps", "cpu").

    Returns:
        torch.device object.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)
