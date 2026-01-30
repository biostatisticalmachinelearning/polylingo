#!/usr/bin/env python3
"""
Train a ResNet classifier for Unicode character classification.

Usage:
    python scripts/train_classifier.py [--epochs N] [--batch-size N] [--lr RATE]

This script trains a ResNet model to classify Unicode characters into their
script/language families using the polylingo package.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from polylingo import (
    ClassifierConfig,
    create_classifier,
    create_data_loaders,
    ClassifierTrainer,
)


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
        description="Train a Unicode character classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/unicode_chars"),
        help="Directory containing the character images",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--model", type=str, default="resnet18",
        choices=["resnet18", "resnet34", "resnet50"],
        help="Model architecture",
    )
    parser.add_argument(
        "--no-pretrained", action="store_true",
        help="Don't use pretrained weights",
    )
    parser.add_argument(
        "--no-augment", action="store_true",
        help="Disable data augmentation",
    )
    parser.add_argument(
        "--no-weighted-sampler", action="store_true",
        help="Disable weighted random sampling",
    )
    parser.add_argument(
        "--no-weighted-loss", action="store_true",
        help="Disable class-weighted loss",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for training",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    set_seed(args.seed)

    # Create configuration
    config = ClassifierConfig(
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

    # Create data loaders
    print("\n" + "=" * 60)
    print("Loading Data")
    print("=" * 60)

    train_loader, test_loader, idx_to_class, class_to_idx, class_weights = create_data_loaders(
        data_dir=config.data_dir,
        mode="classification",
        batch_size=config.batch_size,
        test_split=config.test_split,
        use_balanced_sampling=config.use_weighted_sampler,
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

    model = create_classifier(
        num_classes=config.num_classes,
        model_name=config.model_name,
        pretrained=config.pretrained,
        dropout=config.dropout,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Create trainer
    trainer = ClassifierTrainer(
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
    print(f"Best checkpoint: {config.checkpoint_dir / 'best_model.pt'}")
    print(f"Final Test Accuracy: {100*metrics['accuracy']:.2f}%")


if __name__ == "__main__":
    main()
