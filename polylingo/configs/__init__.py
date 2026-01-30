"""Configuration dataclasses for all models."""

from .base import BaseConfig
from .classifier import ClassifierConfig
from .vae import VAEConfig
from .diffusion import DiffusionConfig

__all__ = [
    "BaseConfig",
    "ClassifierConfig",
    "VAEConfig",
    "DiffusionConfig",
]
