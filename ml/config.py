"""Configuration for the Unicode character classifier."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Training configuration."""

    # Paths
    data_dir: Path = Path("output")
    checkpoint_dir: Path = Path("checkpoints")

    # Data
    image_size: int = 64
    test_split: float = 0.2
    random_seed: int = 42

    # Model
    model_name: str = "resnet18"
    pretrained: bool = True
    num_classes: Optional[int] = None  # Set automatically from data

    # Training
    batch_size: int = 128
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Learning rate scheduler
    lr_scheduler: str = "cosine"  # "cosine" or "step"
    lr_step_size: int = 10
    lr_gamma: float = 0.1

    # Class imbalance handling
    use_weighted_sampler: bool = True
    use_weighted_loss: bool = True
    class_weight_power: float = 0.5  # Smoothing: weight = (1/freq)^power

    # Data augmentation
    use_augmentation: bool = True

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4

    # Hardware
    num_workers: int = 4
    pin_memory: bool = True
    device: str = "auto"  # "auto", "cuda", "mps", or "cpu"

    # Logging
    log_interval: int = 50  # Log every N batches

    def __post_init__(self):
        import torch
        self.data_dir = Path(self.data_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Disable pin_memory on MPS (not supported)
        if self.device == "auto" and torch.backends.mps.is_available():
            self.pin_memory = False
        elif self.device == "mps":
            self.pin_memory = False
