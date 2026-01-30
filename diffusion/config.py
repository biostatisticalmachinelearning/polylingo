"""Configuration for the Diffusion Model."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DiffusionConfig:
    """Diffusion model training configuration."""

    # Paths
    data_dir: Path = Path("output")
    checkpoint_dir: Path = Path("checkpoints_diffusion")

    # Data
    image_size: int = 64
    image_channels: int = 1  # Grayscale
    test_split: float = 0.1
    random_seed: int = 42

    # Model architecture (U-Net)
    base_channels: int = 64
    channel_mults: tuple = (1, 2, 4, 8)
    num_res_blocks: int = 2
    attention_resolutions: tuple = (16, 8)  # Apply attention at these resolutions
    dropout: float = 0.1

    # Conditional generation
    conditional: bool = True  # Condition on script class
    num_classes: Optional[int] = None  # Set from data

    # Diffusion process
    num_timesteps: int = 1000
    beta_schedule: str = "cosine"  # "linear" or "cosine"
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # Training
    batch_size: int = 64
    num_epochs: int = 200
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    ema_decay: float = 0.9999  # Exponential moving average for model weights
    gradient_clip: float = 1.0

    # Class balancing
    use_balanced_sampling: bool = True
    class_weight_power: float = 0.5  # Smoothing for class weights

    # Classifier-free guidance (for conditional generation)
    use_cfg: bool = True  # Classifier-free guidance
    cfg_dropout: float = 0.1  # Probability of dropping class label during training
    cfg_scale: float = 3.0  # Guidance scale during sampling

    # Learning rate scheduler
    lr_scheduler: str = "cosine"
    warmup_steps: int = 1000

    # Hardware
    num_workers: int = 4
    pin_memory: bool = True
    device: str = "auto"

    # Logging and sampling
    log_interval: int = 100
    sample_interval: int = 10  # Generate samples every N epochs
    num_samples: int = 64
    save_interval: int = 20  # Save checkpoint every N epochs

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
