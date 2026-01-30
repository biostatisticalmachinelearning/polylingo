#!/usr/bin/env python3
"""
Train a ResNet classifier for Unicode character classification.

Usage:
    python train.py [--epochs N] [--batch-size N] [--lr RATE] [--no-augment]

This script trains a ResNet18 model to classify Unicode characters into their
script/language families. It handles the imbalanced dataset through:
1. Weighted random sampling during training
2. Class-weighted cross-entropy loss
3. Macro-averaged metrics for fair evaluation across classes
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ml.config import Config
from ml.dataset import create_data_loaders
from ml.model import create_model
from ml.trainer import Trainer


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a Unicode character classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("output"),
        help="Directory containing the character images",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
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
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50"],
        help="Model architecture",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Don't use pretrained weights",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation",
    )
    parser.add_argument(
        "--no-weighted-sampler",
        action="store_true",
        help="Disable weighted random sampling",
    )
    parser.add_argument(
        "--no-weighted-loss",
        action="store_true",
        help="Disable class-weighted loss",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
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
        help="Device to use for training",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Create configuration
    config = Config(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_name=args.model,
        pretrained=not args.no_pretrained,
        use_augmentation=not args.no_augment,
        use_weighted_sampler=not args.no_weighted_sampler,
        use_weighted_loss=not args.no_weighted_loss,
        random_seed=args.seed,
        num_workers=args.workers,
        device=args.device,
    )

    print("=" * 60)
    print("Unicode Character Classifier Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Data directory: {config.data_dir}")
    print(f"  Model: {config.model_name} (pretrained={config.pretrained})")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Augmentation: {config.use_augmentation}")
    print(f"  Weighted sampler: {config.use_weighted_sampler}")
    print(f"  Weighted loss: {config.use_weighted_loss}")
    print(f"  Seed: {config.random_seed}")

    # Create data loaders
    print("\n" + "=" * 60)
    print("Loading Data")
    print("=" * 60)

    train_loader, test_loader, idx_to_class, class_weights = create_data_loaders(config)

    # Create model
    print("\n" + "=" * 60)
    print("Creating Model")
    print("=" * 60)

    model = create_model(
        num_classes=config.num_classes,
        model_name=config.model_name,
        pretrained=config.pretrained,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        class_weights=class_weights,
        idx_to_class=idx_to_class,
    )

    # Train
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)

    history = trainer.train(train_loader, test_loader)

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)

    # Load best model
    trainer.load_checkpoint("best_model.pt")
    metrics = trainer.print_classification_report(test_loader)

    # Save training history
    history_path = config.checkpoint_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to: {history_path}")

    # Save class mapping
    mapping_path = config.checkpoint_dir / "class_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump({str(k): v for k, v in idx_to_class.items()}, f, indent=2)
    print(f"Class mapping saved to: {mapping_path}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best checkpoint saved to: {config.checkpoint_dir / 'best_model.pt'}")
    print(f"Final Test Accuracy: {100*metrics['accuracy']:.2f}%")
    print(f"Final Test F1 (macro): {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
