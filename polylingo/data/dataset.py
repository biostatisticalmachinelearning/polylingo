"""Unified dataset class for Unicode character images."""

import json
from pathlib import Path
from typing import Optional, Callable, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

# Default dataset filters used across training/eval scripts.
DEFAULT_INCLUDE_SCRIPTS: tuple[str, ...] = (
    "han_cjk",
    "hiragana",
    "katakana",
    "hangul",
)
DEFAULT_EXCLUDE_SCRIPTS: tuple[str, ...] = ("symbols",)


class UnicodeDataset(Dataset):
    """Dataset for Unicode character images.

    Supports multiple modes:
    - classification: returns (image, label)
    - reconstruction: returns image only (for VAE)
    - generation: returns (image, label) with normalized images for diffusion

    Args:
        image_paths: List of paths to image files.
        labels: List of integer class labels.
        transform: Image transforms to apply.
        mode: One of "classification", "reconstruction", "generation".
        return_path: If True, also return the image path.
    """

    def __init__(
        self,
        image_paths: list[Path],
        labels: list[int],
        transform: Optional[Callable] = None,
        mode: str = "classification",
        return_path: bool = False,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.mode = mode
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        if self.mode == "classification":
            image = Image.open(image_path).convert("RGB")
        else:
            # Grayscale for VAE/diffusion
            image = Image.open(image_path).convert("L")

        if self.transform:
            image = self.transform(image)

        # Build return value based on mode
        if self.mode == "reconstruction":
            if self.return_path:
                return image, str(image_path)
            return image
        else:
            if self.return_path:
                return image, label, str(image_path)
            return image, label


def _normalize_script_filter(scripts: Optional[Sequence[str]]) -> set[str]:
    """Normalize script names from user/config inputs."""
    if scripts is None:
        return set()
    return {name.strip() for name in scripts if name and name.strip()}


def load_dataset_info(
    data_dir: Path,
    include_scripts: Optional[Sequence[str]] = None,
    exclude_scripts: Optional[Sequence[str]] = None,
) -> tuple[list[Path], list[int], dict[int, str], dict[str, int]]:
    """Load all image paths and labels from a dataset directory.

    Args:
        data_dir: Path to directory containing script subdirectories.
        include_scripts: Optional script allowlist. Defaults to CJK scripts.
        exclude_scripts: Optional script denylist. Defaults to excluding symbols.

    Returns:
        Tuple of (image_paths, labels, idx_to_class, class_to_idx).
    """
    data_dir = Path(data_dir)

    if include_scripts is None:
        include_scripts = DEFAULT_INCLUDE_SCRIPTS
    if exclude_scripts is None:
        exclude_scripts = DEFAULT_EXCLUDE_SCRIPTS

    include_set = _normalize_script_filter(include_scripts)
    exclude_set = _normalize_script_filter(exclude_scripts)

    # Get all script directories (sorted for reproducibility)
    all_script_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])
    script_dirs = [
        d for d in all_script_dirs
        if (not include_set or d.name in include_set) and d.name not in exclude_set
    ]

    if not script_dirs:
        available_scripts = ", ".join(d.name for d in all_script_dirs) or "<none>"
        include_label = ", ".join(sorted(include_set)) if include_set else "<all>"
        exclude_label = ", ".join(sorted(exclude_set)) if exclude_set else "<none>"
        raise ValueError(
            "No scripts found after applying filters. "
            f"include={include_label}, exclude={exclude_label}, available={available_scripts}"
        )

    # Build class mappings
    class_names = [d.name for d in script_dirs]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    # Collect all image paths and labels
    image_paths = []
    labels = []

    for script_dir in script_dirs:
        class_idx = class_to_idx[script_dir.name]
        for image_path in sorted(script_dir.glob("*.png")):
            image_paths.append(image_path)
            labels.append(class_idx)

    return image_paths, labels, idx_to_class, class_to_idx


def load_metadata(data_dir: Path) -> dict:
    """Load metadata.json from dataset directory."""
    metadata_path = Path(data_dir) / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f)
    return {}
