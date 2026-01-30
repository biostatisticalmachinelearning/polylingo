"""Configuration for the Variational Autoencoder."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .base import BaseConfig


@dataclass
class VAEConfig(BaseConfig):
    """VAE training configuration."""

    # Paths
    checkpoint_dir: Path = Path("checkpoints/vae")

    # Model architecture
    latent_dim: int = 64  # Size of latent space
    hidden_dims: tuple = (32, 64, 128, 256)  # Conv channel sizes

    # Beta-VAE: higher beta = more disentangled but worse reconstruction
    beta: float = 1.0  # KL divergence weight (1.0 = standard VAE)

    # Conditional VAE (condition on script label)
    conditional: bool = False
    num_classes: Optional[int] = None  # Set from data if conditional

    # Training
    num_epochs: int = 100
    weight_decay: float = 1e-5

    # Learning rate scheduler
    lr_patience: int = 5  # For plateau scheduler

    # Early stopping
    patience: int = 15

    # Visualization
    sample_interval: int = 5  # Generate samples every N epochs
    num_samples: int = 64  # Number of samples to generate
