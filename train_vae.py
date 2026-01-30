#!/usr/bin/env python3
"""
Train a Variational Autoencoder on Unicode character images.

Usage:
    python train_vae.py [--latent-dim N] [--beta FLOAT] [--epochs N]

This script trains a convolutional VAE to learn a latent representation
of Unicode characters. The learned latent space can be explored to
understand how different scripts and characters are encoded.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from vae.config import VAEConfig
from vae.dataset import create_vae_data_loaders
from vae.model import create_vae
from vae.trainer import VAETrainer


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a VAE on Unicode character images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("output"),
        help="Directory containing the character images",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=64,
        help="Dimension of the latent space",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Beta parameter for KL divergence weight (beta-VAE)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256],
        help="Hidden dimensions for encoder/decoder",
    )
    parser.add_argument(
        "--conditional",
        action="store_true",
        help="Train a conditional VAE (condition on script label)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Create configuration
    config = VAEConfig(
        data_dir=args.data_dir,
        latent_dim=args.latent_dim,
        beta=args.beta,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dims=tuple(args.hidden_dims),
        conditional=args.conditional,
        random_seed=args.seed,
        num_workers=args.workers,
        device=args.device,
    )

    print("=" * 60)
    print("Unicode Character VAE Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Data directory: {config.data_dir}")
    print(f"  Latent dimension: {config.latent_dim}")
    print(f"  Beta (KL weight): {config.beta}")
    print(f"  Hidden dims: {config.hidden_dims}")
    print(f"  Conditional: {config.conditional}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")

    # Create data loaders
    print("\n" + "=" * 60)
    print("Loading Data")
    print("=" * 60)

    train_loader, test_loader, idx_to_class, class_to_idx = create_vae_data_loaders(config)

    # Create model
    print("\n" + "=" * 60)
    print("Creating Model")
    print("=" * 60)

    model = create_vae(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Latent space dimension: {config.latent_dim}")

    # Create trainer
    trainer = VAETrainer(
        model=model,
        config=config,
        idx_to_class=idx_to_class,
    )

    # Train
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    history = trainer.train(train_loader, test_loader)

    # Save final model and history
    trainer.save_checkpoint("final_vae.pt")

    history_path = config.checkpoint_dir / "vae_training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to: {history_path}")

    # Save class mapping
    mapping_path = config.checkpoint_dir / "vae_class_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump({str(k): v for k, v in idx_to_class.items()}, f, indent=2)
    print(f"Class mapping saved to: {mapping_path}")

    # Generate final samples
    print("\n" + "=" * 60)
    print("Generating Final Samples")
    print("=" * 60)

    trainer.load_checkpoint("best_vae.pt")
    trainer.generate_samples(999, test_loader)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best checkpoint: {config.checkpoint_dir / 'best_vae.pt'}")
    print(f"Samples directory: {trainer.samples_dir}")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
