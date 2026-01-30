"""Class balancing utilities for imbalanced datasets."""

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def compute_class_weights(
    labels: list[int],
    num_classes: int,
    power: float = 0.5,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute class weights inversely proportional to frequency.

    Uses smoothed inverse frequency: weight = (N / count)^power
    where power < 1 reduces the impact of very rare classes.

    Args:
        labels: List of integer class labels.
        num_classes: Total number of classes.
        power: Exponent for smoothing (0.5 = square root).
        normalize: If True, normalize so mean weight is 1.

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
    if normalize:
        weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


def compute_sample_weights(
    labels: list[int],
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute per-sample weights from class weights.

    Args:
        labels: List of integer class labels.
        class_weights: Tensor of class weights.

    Returns:
        Tensor of sample weights.
    """
    return torch.tensor([class_weights[label].item() for label in labels])


def create_balanced_sampler(
    labels: list[int],
    class_weights: torch.Tensor,
    num_samples: int = None,
) -> WeightedRandomSampler:
    """Create a weighted random sampler for balanced training.

    Args:
        labels: List of integer class labels.
        class_weights: Tensor of class weights.
        num_samples: Number of samples per epoch (default: len(labels)).

    Returns:
        WeightedRandomSampler instance.
    """
    sample_weights = compute_sample_weights(labels, class_weights)

    if num_samples is None:
        num_samples = len(labels)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True,
    )


def print_class_distribution(
    labels: list[int],
    idx_to_class: dict[int, str],
    class_weights: torch.Tensor = None,
    max_show: int = 15,
):
    """Print class distribution and optionally weights.

    Args:
        labels: List of integer class labels.
        idx_to_class: Mapping from indices to class names.
        class_weights: Optional tensor of class weights.
        max_show: Maximum number of classes to show.
    """
    num_classes = len(idx_to_class)
    counts = np.bincount(labels, minlength=num_classes)

    print(f"\nClass distribution ({num_classes} classes, {len(labels)} samples):")

    # Sort by count (descending)
    sorted_indices = np.argsort(counts)[::-1]

    for i, idx in enumerate(sorted_indices[:max_show]):
        class_name = idx_to_class[idx]
        count = counts[idx]
        weight_str = f", weight={class_weights[idx]:.3f}" if class_weights is not None else ""
        print(f"  {class_name:25s}: {count:5d}{weight_str}")

    if num_classes > max_show:
        print(f"  ... and {num_classes - max_show} more classes")

    if class_weights is not None:
        print(f"\nWeight range: [{class_weights.min():.3f}, {class_weights.max():.3f}]")
        print(f"Weight ratio: {class_weights.max() / class_weights.min():.2f}x")
