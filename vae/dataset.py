"""Dataset and data loading utilities for VAE training."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .config import VAEConfig


class UnicodeVAEDataset(Dataset):
    """Dataset for VAE training on Unicode character images."""

    def __init__(
        self,
        image_paths: list[Path],
        labels: list[int],
        transform: Optional[transforms.Compose] = None,
        return_labels: bool = False,
    ):
        """
        Args:
            image_paths: List of paths to image files.
            labels: List of integer class labels.
            transform: Optional torchvision transforms.
            return_labels: Whether to return labels (for conditional VAE).
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.return_labels = return_labels

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image as grayscale
        image = Image.open(image_path).convert("L")

        if self.transform:
            image = self.transform(image)

        if self.return_labels:
            return image, label
        return image


def get_vae_transforms(config: VAEConfig, train: bool = True) -> transforms.Compose:
    """Get image transforms for VAE training.

    Args:
        config: VAE configuration.
        train: Whether to include training augmentations.

    Returns:
        Composed transforms.
    """
    transform_list = []

    if train:
        # Light augmentation for VAE (we want to preserve character structure)
        transform_list.extend([
            transforms.RandomAffine(
                degrees=5,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),
        ])

    transform_list.extend([
        transforms.ToTensor(),
        # Note: No normalization - we want [0, 1] for BCE loss
    ])

    return transforms.Compose(transform_list)


def load_vae_dataset(config: VAEConfig) -> tuple[dict, dict[int, str], dict[str, int]]:
    """Load all image paths and labels from the data directory.

    Args:
        config: VAE configuration.

    Returns:
        Tuple of (data_dict, idx_to_class, class_to_idx).
    """
    data_dir = config.data_dir

    # Get all script directories
    script_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir()
    ])

    # Build class mappings
    class_names = [d.name for d in script_dirs]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    # Collect all image paths and labels
    image_paths = []
    labels = []

    for script_dir in script_dirs:
        class_idx = class_to_idx[script_dir.name]
        for image_path in script_dir.glob("*.png"):
            image_paths.append(image_path)
            labels.append(class_idx)

    print(f"Loaded {len(image_paths)} images across {len(class_names)} classes")

    return (
        {"paths": image_paths, "labels": labels},
        idx_to_class,
        class_to_idx,
    )


def create_vae_data_loaders(config: VAEConfig) -> tuple[
    DataLoader,
    DataLoader,
    dict[int, str],
    dict[str, int],
]:
    """Create train and test data loaders for VAE.

    Args:
        config: VAE configuration.

    Returns:
        Tuple of (train_loader, test_loader, idx_to_class, class_to_idx).
    """
    # Load all data
    data_dict, idx_to_class, class_to_idx = load_vae_dataset(config)
    paths = data_dict["paths"]
    labels = data_dict["labels"]

    num_classes = len(idx_to_class)
    config.num_classes = num_classes

    # Stratified train/test split
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths,
        labels,
        test_size=config.test_split,
        random_state=config.random_seed,
        stratify=labels,
    )

    print(f"Train set: {len(train_paths)} samples")
    print(f"Test set: {len(test_paths)} samples")

    # Create transforms
    train_transform = get_vae_transforms(config, train=True)
    test_transform = get_vae_transforms(config, train=False)

    # Create datasets
    train_dataset = UnicodeVAEDataset(
        train_paths, train_labels, train_transform,
        return_labels=config.conditional,
    )
    test_dataset = UnicodeVAEDataset(
        test_paths, test_labels, test_transform,
        return_labels=config.conditional,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return train_loader, test_loader, idx_to_class, class_to_idx
