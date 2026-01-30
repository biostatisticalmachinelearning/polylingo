"""Configuration for the Variational Autoencoder."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class VAEConfig:
    """VAE training configuration."""

    # Paths
    data_dir: Path = Path("output")
    checkpoint_dir: Path = Path("checkpoints_vae")

    # Data
    image_size: int = 64
    image_channels: int = 1  # Grayscale for characters
    test_split: float = 0.2
    random_seed: int = 42

    # Model architecture
    latent_dim: int = 64  # Size of latent space
    hidden_dims: tuple = (32, 64, 128, 256)  # Conv channel sizes

    # Beta-VAE: higher beta = more disentangled but worse reconstruction
    beta: float = 1.0  # KL divergence weight (1.0 = standard VAE)

    # Conditional VAE (condition on script label)
    conditional: bool = False
    num_classes: Optional[int] = None  # Set from data if conditional

    # Training
    batch_size: int = 128
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # Learning rate scheduler
    lr_scheduler: str = "cosine"  # "cosine", "step", or "plateau"
    lr_patience: int = 5  # For plateau scheduler

    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4

    # Hardware
    num_workers: int = 4
    pin_memory: bool = True
    device: str = "auto"

    # Logging and visualization
    log_interval: int = 50
    sample_interval: int = 5  # Generate samples every N epochs
    num_samples: int = 64  # Number of samples to generate

    def __post_init__(self):
        import torch
        self.data_dir = Path(self.data_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Disable pin_memory on MPS
        if self.device == "auto" and torch.backends.mps.is_available():
            self.pin_memory = False
        elif self.device == "mps":
            self.pin_memory = False
