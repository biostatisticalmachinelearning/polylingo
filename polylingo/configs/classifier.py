"""Configuration for the ResNet classifier."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .base import BaseConfig


@dataclass
class ClassifierConfig(BaseConfig):
    """Classifier training configuration."""

    # Paths
    checkpoint_dir: Path = Path("checkpoints/classifier")

    # Model
    model_name: str = "resnet18"  # "resnet18", "resnet34", "resnet50"
    pretrained: bool = True
    dropout: float = 0.2
    num_classes: Optional[int] = None  # Set automatically from data

    # Class imbalance handling
    use_weighted_sampler: bool = True
    use_weighted_loss: bool = True
    class_weight_power: float = 0.5  # Smoothing: weight = (1/freq)^power

    # Data augmentation
    use_augmentation: bool = True

    # Learning rate scheduler
    lr_step_size: int = 10
    lr_gamma: float = 0.1
