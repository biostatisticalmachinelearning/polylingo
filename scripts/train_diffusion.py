#!/usr/bin/env python3
"""
Train a Diffusion Model for Unicode character generation.

Usage:
    python scripts/train_diffusion.py [--epochs N] [--batch-size N] [--timesteps N]

This script trains a conditional diffusion model (DDPM) that can generate
new Unicode character images using the polylingo package.

Features:
- Balanced sampling to ensure minority scripts are well-represented
- Classifier-free guidance for conditional generation
- DDIM sampling for faster inference
- EMA for stable training
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from polylingo import (
    DiffusionConfig,
    SimpleUNet,
    GaussianDiffusion,
    create_data_loaders,
    DiffusionTrainer,
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a diffusion model for Unicode character generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/unicode_chars"),
        help="Directory containing character images",
    )
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000,
        help="Number of diffusion timesteps",
    )
    parser.add_argument(
        "--base-channels", type=int, default=64,
        help="Base channel count for U-Net",
    )
    parser.add_argument(
        "--beta-schedule", type=str, default="cosine",
        choices=["linear", "cosine"],
        help="Noise schedule type",
    )
    parser.add_argument(
        "--cfg-scale", type=float, default=3.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--no-cfg", action="store_true",
        help="Disable classifier-free guidance",
    )
    parser.add_argument(
        "--no-balanced-sampling", action="store_true",
        help="Disable balanced class sampling",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--sample-interval", type=int, default=10,
        help="Generate samples every N epochs",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    set_seed(args.seed)

    # Create configuration
    config = DiffusionConfig(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_timesteps=args.timesteps,
        base_channels=args.base_channels,
        beta_schedule=args.beta_schedule,
        use_cfg=not args.no_cfg,
        cfg_scale=args.cfg_scale,
        use_balanced_sampling=not args.no_balanced_sampling,
        random_seed=args.seed,
        num_workers=args.workers,
        device=args.device,
        sample_interval=args.sample_interval,
    )

    print("=" * 60)
    print("Unicode Character Diffusion Model Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Data directory: {config.data_dir}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Timesteps: {config.num_timesteps}")
    print(f"  Beta schedule: {config.beta_schedule}")
    print(f"  Classifier-free guidance: {config.use_cfg}")
    print(f"  CFG scale: {config.cfg_scale}")
    print(f"  Balanced sampling: {config.use_balanced_sampling}")

    # Create data loaders
    print("\n" + "=" * 60)
    print("Loading Data")
    print("=" * 60)

    train_loader, test_loader, idx_to_class, class_to_idx, class_weights = create_data_loaders(
        data_dir=config.data_dir,
        mode="diffusion",
        batch_size=config.batch_size,
        test_split=config.test_split,
        use_balanced_sampling=config.use_balanced_sampling,
        class_weight_power=config.class_weight_power,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        random_seed=config.random_seed,
    )

    config.num_classes = len(idx_to_class)

    # Create model
    print("\n" + "=" * 60)
    print("Creating Model")
    print("=" * 60)

    model = SimpleUNet(
        image_channels=config.image_channels,
        base_channels=config.base_channels,
        channel_mults=config.channel_mults,
        dropout=config.dropout,
        num_classes=config.num_classes if config.conditional else None,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net parameters: {total_params:,}")

    # Create diffusion process
    diffusion = GaussianDiffusion(
        model=model,
        image_size=config.image_size,
        image_channels=config.image_channels,
        timesteps=config.num_timesteps,
        beta_schedule=config.beta_schedule,
    )

    # Create trainer
    trainer = DiffusionTrainer(
        diffusion=diffusion,
        config=config,
        idx_to_class=idx_to_class,
    )

    # Train
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    history = trainer.train(train_loader)

    # Save final model and history
    trainer.save_checkpoint("final_diffusion.pt")

    history_path = config.checkpoint_dir / "diffusion_training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to: {history_path}")

    # Save class mapping
    mapping_path = config.checkpoint_dir / "diffusion_class_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump({str(k): v for k, v in idx_to_class.items()}, f, indent=2)
    print(f"Class mapping saved to: {mapping_path}")

    # Generate final samples
    print("\n" + "=" * 60)
    print("Generating Final Samples")
    print("=" * 60)

    trainer.load_checkpoint("best_diffusion.pt")
    trainer.generate_samples(999, num_samples=64, use_ddim=True, ddim_steps=50)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best checkpoint: {config.checkpoint_dir / 'best_diffusion.pt'}")
    print(f"Samples directory: {trainer.samples_dir}")
    print(f"Best training loss: {trainer.best_loss:.4f}")


if __name__ == "__main__":
    main()
