"""Base configuration shared by all models."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch


@dataclass
class BaseConfig:
    """Base configuration with common settings."""

    # Paths
    data_dir: Path = Path("data/unicode_chars")
    checkpoint_dir: Path = Path("checkpoints")

    # Data
    image_size: int = 64
    image_channels: int = 1  # Grayscale
    test_split: float = 0.2
    random_seed: int = 42

    # Training
    batch_size: int = 128
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Learning rate scheduler
    lr_scheduler: str = "cosine"

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4

    # Hardware
    num_workers: int = 4
    pin_memory: bool = True
    device: str = "auto"

    # Logging
    log_interval: int = 50

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Disable pin_memory on MPS (not supported)
        if self.device == "auto" and torch.backends.mps.is_available():
            self.pin_memory = False
        elif self.device == "mps":
            self.pin_memory = False
