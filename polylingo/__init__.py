"""
Polylingo: Machine Learning for Unicode Characters

A toolkit for analyzing and generating Unicode characters across
world writing systems using deep learning.
"""

__version__ = "0.1.0"

# Import main components for convenient access
from .data import (
    UnicodeDataset,
    create_data_loaders,
    load_dataset_info,
    DEFAULT_INCLUDE_SCRIPTS,
    DEFAULT_EXCLUDE_SCRIPTS,
)

from .models import (
    create_classifier,
    ConvVAE,
    ConditionalConvVAE,
    vae_loss,
    SimpleUNet,
    GaussianDiffusion,
)

from .configs import (
    BaseConfig,
    ClassifierConfig,
    VAEConfig,
    DiffusionConfig,
)

from .trainers import (
    ClassifierTrainer,
    VAETrainer,
    DiffusionTrainer,
)

from .utils import (
    get_device,
    EMA,
    EarlyStopping,
)

__all__ = [
    # Data
    "UnicodeDataset",
    "create_data_loaders",
    "load_dataset_info",
    "DEFAULT_INCLUDE_SCRIPTS",
    "DEFAULT_EXCLUDE_SCRIPTS",
    # Models
    "create_classifier",
    "ConvVAE",
    "ConditionalConvVAE",
    "vae_loss",
    "SimpleUNet",
    "GaussianDiffusion",
    # Configs
    "BaseConfig",
    "ClassifierConfig",
    "VAEConfig",
    "DiffusionConfig",
    # Trainers
    "ClassifierTrainer",
    "VAETrainer",
    "DiffusionTrainer",
    # Utils
    "get_device",
    "EMA",
    "EarlyStopping",
]
