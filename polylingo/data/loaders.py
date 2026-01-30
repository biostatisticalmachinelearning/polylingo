"""Data loader creation utilities."""

from pathlib import Path
from typing import Optional

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .dataset import UnicodeDataset, load_dataset_info
from .transforms import get_transforms
from .balancing import (
    compute_class_weights,
    create_balanced_sampler,
    print_class_distribution,
)


def create_data_loaders(
    data_dir: Path,
    mode: str = "classification",
    batch_size: int = 64,
    test_split: float = 0.2,
    use_balanced_sampling: bool = True,
    class_weight_power: float = 0.5,
    num_workers: int = 4,
    pin_memory: bool = True,
    random_seed: int = 42,
    image_size: int = 64,
    verbose: bool = True,
) -> tuple[DataLoader, DataLoader, dict[int, str], dict[str, int], Optional[torch.Tensor]]:
    """Create train and test data loaders.

    This is the main entry point for creating data loaders for any model type.

    Args:
        data_dir: Path to dataset directory.
        mode: One of "classification", "vae", "diffusion".
        batch_size: Batch size for data loaders.
        test_split: Fraction of data for test set.
        use_balanced_sampling: Whether to use weighted random sampling.
        class_weight_power: Power for class weight computation.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory (disable for MPS).
        random_seed: Random seed for reproducibility.
        image_size: Target image size.
        verbose: Whether to print statistics.

    Returns:
        Tuple of (train_loader, test_loader, idx_to_class, class_to_idx, class_weights).
    """
    # Check for MPS and disable pin_memory if needed
    if torch.backends.mps.is_available():
        pin_memory = False

    # Load dataset info
    paths, labels, idx_to_class, class_to_idx = load_dataset_info(data_dir)
    num_classes = len(idx_to_class)

    if verbose:
        print(f"Loaded {len(paths)} images across {num_classes} classes")

    # Stratified train/test split
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths,
        labels,
        test_size=test_split,
        random_state=random_seed,
        stratify=labels,
    )

    if verbose:
        print(f"Train set: {len(train_paths)} samples")
        print(f"Test set: {len(test_paths)} samples")

    # Compute class weights
    class_weights = compute_class_weights(
        train_labels, num_classes, power=class_weight_power
    )

    if verbose:
        print_class_distribution(train_labels, idx_to_class, class_weights)

    # Get transforms based on mode
    dataset_mode = "classification" if mode == "classification" else "reconstruction"
    if mode == "diffusion":
        dataset_mode = "generation"

    train_transform = get_transforms(mode=mode, train=True, image_size=image_size)
    test_transform = get_transforms(mode=mode, train=False, image_size=image_size)

    # Create datasets
    train_dataset = UnicodeDataset(
        train_paths, train_labels, train_transform,
        mode=dataset_mode if mode != "diffusion" else "generation",
    )
    test_dataset = UnicodeDataset(
        test_paths, test_labels, test_transform,
        mode=dataset_mode if mode != "diffusion" else "generation",
    )

    # Create sampler for balanced training
    sampler = None
    shuffle = True
    if use_balanced_sampling:
        sampler = create_balanced_sampler(train_labels, class_weights)
        shuffle = False
        if verbose:
            print("\nUsing weighted random sampling for balanced training")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader, idx_to_class, class_to_idx, class_weights
