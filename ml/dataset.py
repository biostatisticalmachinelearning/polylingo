"""Dataset and data loading utilities for Unicode character classification."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from .config import Config


class UnicodeCharacterDataset(Dataset):
    """Dataset for Unicode character images."""

    def __init__(
        self,
        image_paths: list[Path],
        labels: list[int],
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            image_paths: List of paths to image files.
            labels: List of integer class labels.
            transform: Optional torchvision transforms.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(config: Config, train: bool = True) -> transforms.Compose:
    """Get image transforms for training or evaluation.

    Args:
        config: Configuration object.
        train: Whether to include training augmentations.

    Returns:
        Composed transforms.
    """
    transform_list = []

    if train and config.use_augmentation:
        transform_list.extend([
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
            ),
            transforms.RandomInvert(p=0.1),  # Occasionally invert colors
        ])

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return transforms.Compose(transform_list)


def load_dataset(config: Config) -> tuple[
    dict[str, list],
    dict[int, str],
    dict[str, int],
]:
    """Load all image paths and labels from the data directory.

    Args:
        config: Configuration object.

    Returns:
        Tuple of (data_dict, idx_to_class, class_to_idx) where:
            - data_dict has keys 'paths' and 'labels'
            - idx_to_class maps integer indices to class names
            - class_to_idx maps class names to integer indices
    """
    data_dir = config.data_dir

    # Get all script directories (excluding metadata.json)
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


def compute_class_weights(
    labels: list[int],
    num_classes: int,
    power: float = 0.5,
) -> torch.Tensor:
    """Compute class weights inversely proportional to class frequency.

    Uses smoothed inverse frequency: weight = (N / count)^power
    where power < 1 reduces the impact of very rare classes.

    Args:
        labels: List of integer class labels.
        num_classes: Total number of classes.
        power: Exponent for smoothing (0.5 = square root).

    Returns:
        Tensor of class weights.
    """
    labels_array = np.array(labels)
    class_counts = np.bincount(labels_array, minlength=num_classes).astype(float)

    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)

    # Inverse frequency with smoothing
    total = len(labels)
    weights = (total / class_counts) ** power

    # Normalize so mean weight is 1
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


def compute_sample_weights(
    labels: list[int],
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute per-sample weights for WeightedRandomSampler.

    Args:
        labels: List of integer class labels.
        class_weights: Tensor of class weights.

    Returns:
        Tensor of sample weights.
    """
    return torch.tensor([class_weights[label].item() for label in labels])


def create_data_loaders(config: Config) -> tuple[
    DataLoader,
    DataLoader,
    dict[int, str],
    torch.Tensor,
]:
    """Create train and test data loaders with stratified split.

    Args:
        config: Configuration object.

    Returns:
        Tuple of (train_loader, test_loader, idx_to_class, class_weights).
    """
    # Load all data
    data_dict, idx_to_class, class_to_idx = load_dataset(config)
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

    # Print class distribution
    print("\nClass distribution:")
    train_counts = np.bincount(train_labels, minlength=num_classes)
    test_counts = np.bincount(test_labels, minlength=num_classes)

    for idx in range(num_classes):
        class_name = idx_to_class[idx]
        print(f"  {class_name:25s}: train={train_counts[idx]:5d}, test={test_counts[idx]:4d}")

    # Compute class weights for loss function
    class_weights = compute_class_weights(
        train_labels,
        num_classes,
        power=config.class_weight_power,
    )

    # Create transforms
    train_transform = get_transforms(config, train=True)
    test_transform = get_transforms(config, train=False)

    # Create datasets
    train_dataset = UnicodeCharacterDataset(
        train_paths, train_labels, train_transform
    )
    test_dataset = UnicodeCharacterDataset(
        test_paths, test_labels, test_transform
    )

    # Create sampler for imbalanced data
    sampler = None
    shuffle = True
    if config.use_weighted_sampler:
        sample_weights = compute_sample_weights(train_labels, class_weights)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        shuffle = False  # Sampler handles shuffling

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
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

    return train_loader, test_loader, idx_to_class, class_weights
