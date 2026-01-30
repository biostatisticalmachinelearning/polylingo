"""Dataset and data loading for diffusion model training with balanced sampling."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from .config import DiffusionConfig


class UnicodeCharacterDataset(Dataset):
    """Dataset for Unicode character images."""

    def __init__(
        self,
        image_paths: list[Path],
        labels: list[int],
        transform: Optional[transforms.Compose] = None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image as grayscale
        image = Image.open(image_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(train: bool = True) -> transforms.Compose:
    """Get image transforms."""
    transform_list = []

    if train:
        # Data augmentation
        transform_list.extend([
            transforms.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
            ),
            transforms.RandomHorizontalFlip(p=0.01),  # Very rare - most scripts are directional
        ])

    transform_list.extend([
        transforms.ToTensor(),
        # Scale to [-1, 1] for diffusion
        transforms.Normalize([0.5], [0.5]),
    ])

    return transforms.Compose(transform_list)


def compute_class_weights(
    labels: list[int],
    num_classes: int,
    power: float = 0.5,
) -> torch.Tensor:
    """Compute class weights inversely proportional to frequency.

    Uses smoothed inverse frequency: weight = (N / count)^power
    """
    labels_array = np.array(labels)
    class_counts = np.bincount(labels_array, minlength=num_classes).astype(float)
    class_counts = np.maximum(class_counts, 1)  # Avoid division by zero

    total = len(labels)
    weights = (total / class_counts) ** power
    weights = weights / weights.mean()  # Normalize

    return torch.tensor(weights, dtype=torch.float32)


def load_dataset(config: DiffusionConfig) -> tuple[dict, dict[int, str], dict[str, int]]:
    """Load all image paths and labels."""
    data_dir = config.data_dir

    script_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in script_dirs]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    image_paths = []
    labels = []

    for script_dir in script_dirs:
        class_idx = class_to_idx[script_dir.name]
        for image_path in script_dir.glob("*.png"):
            image_paths.append(image_path)
            labels.append(class_idx)

    print(f"Loaded {len(image_paths)} images across {len(class_names)} classes")

    return {"paths": image_paths, "labels": labels}, idx_to_class, class_to_idx


def create_diffusion_data_loaders(config: DiffusionConfig) -> tuple[
    DataLoader,
    DataLoader,
    dict[int, str],
    dict[str, int],
    torch.Tensor,
]:
    """Create balanced data loaders for diffusion training.

    Returns:
        Tuple of (train_loader, test_loader, idx_to_class, class_to_idx, class_weights).
    """
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

    # Compute class weights for balanced sampling
    class_weights = compute_class_weights(
        train_labels, num_classes, power=config.class_weight_power
    )

    # Print class distribution and weights
    print("\nClass distribution and sampling weights:")
    train_counts = np.bincount(train_labels, minlength=num_classes)
    for idx in range(min(10, num_classes)):  # Show first 10
        class_name = idx_to_class[idx]
        print(f"  {class_name:25s}: {train_counts[idx]:5d} samples, weight={class_weights[idx]:.3f}")
    if num_classes > 10:
        print(f"  ... and {num_classes - 10} more classes")

    # Show weight statistics
    print(f"\nWeight statistics:")
    print(f"  Min weight: {class_weights.min():.3f} (most common class)")
    print(f"  Max weight: {class_weights.max():.3f} (least common class)")
    print(f"  Weight ratio: {class_weights.max() / class_weights.min():.2f}x")

    # Create transforms
    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)

    # Create datasets
    train_dataset = UnicodeCharacterDataset(train_paths, train_labels, train_transform)
    test_dataset = UnicodeCharacterDataset(test_paths, test_labels, test_transform)

    # Create balanced sampler
    sampler = None
    shuffle = True
    if config.use_balanced_sampling:
        sample_weights = torch.tensor([class_weights[label].item() for label in train_labels])
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        shuffle = False
        print("\nUsing weighted random sampling for balanced training")

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

    return train_loader, test_loader, idx_to_class, class_to_idx, class_weights
