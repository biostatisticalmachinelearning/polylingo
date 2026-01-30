#!/usr/bin/env python3
"""
Explore the latent space of a trained VAE.

This script provides various tools for understanding how Unicode characters
are represented in the VAE's latent space:

1. Latent space visualization (t-SNE, UMAP, PCA)
2. Latent dimension traversal
3. Character interpolation
4. Nearest neighbor analysis
5. Random sampling

Usage:
    python explore_latent.py visualize --method tsne
    python explore_latent.py traverse --dim 0 --steps 10
    python explore_latent.py interpolate --char1 output/latin/0041.png --char2 output/latin/005A.png
    python explore_latent.py neighbors --char output/latin/0041.png --k 10
    python explore_latent.py sample --num 64
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from vae.model import create_vae
from vae.trainer import get_device


def load_model(checkpoint_path: Path, device: torch.device):
    """Load a trained VAE model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    idx_to_class = checkpoint.get("idx_to_class", {})

    model = create_vae(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config, idx_to_class


def get_transform():
    """Get image transform for loading."""
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])


def load_image(path: Path, transform, device: torch.device) -> torch.Tensor:
    """Load and transform a single image."""
    image = Image.open(path).convert("L")
    return transform(image).unsqueeze(0).to(device)


def collect_latents(
    model,
    data_dir: Path,
    device: torch.device,
    max_per_class: Optional[int] = None,
) -> tuple[np.ndarray, list[int], list[str], list[Path]]:
    """Collect latent representations for all images.

    Returns:
        Tuple of (latents, labels, class_names, paths).
    """
    transform = get_transform()

    # Get all script directories
    script_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in script_dirs]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    latents = []
    labels = []
    paths = []

    print("Collecting latent representations...")
    for script_dir in tqdm(script_dirs):
        class_idx = class_to_idx[script_dir.name]
        image_paths = list(script_dir.glob("*.png"))

        if max_per_class and len(image_paths) > max_per_class:
            # Random sample
            indices = np.random.choice(len(image_paths), max_per_class, replace=False)
            image_paths = [image_paths[i] for i in indices]

        for image_path in image_paths:
            image = load_image(image_path, transform, device)
            with torch.no_grad():
                z = model.get_latent(image)
            latents.append(z.cpu().numpy().squeeze())
            labels.append(class_idx)
            paths.append(image_path)

    return np.array(latents), labels, class_names, paths


def visualize_latent_space(
    model,
    config,
    idx_to_class: dict,
    data_dir: Path,
    device: torch.device,
    method: str = "tsne",
    output_path: Optional[Path] = None,
    max_per_class: int = 100,
):
    """Visualize the latent space using dimensionality reduction."""
    latents, labels, class_names, _ = collect_latents(
        model, data_dir, device, max_per_class
    )

    print(f"Reducing dimensionality with {method.upper()}...")

    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        embedded = reducer.fit_transform(latents)
    elif method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            embedded = reducer.fit_transform(latents)
        except ImportError:
            print("UMAP not installed. Install with: pip install umap-learn")
            return
    elif method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        embedded = reducer.fit_transform(latents)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Plot
    plt.figure(figsize=(16, 12))

    # Use a colormap with enough colors
    cmap = plt.cm.get_cmap('tab20', len(class_names))
    colors = [cmap(labels[i] % 20) for i in range(len(labels))]

    scatter = plt.scatter(
        embedded[:, 0], embedded[:, 1],
        c=labels, cmap='tab20', alpha=0.6, s=10
    )

    plt.title(f"VAE Latent Space ({method.upper()}) - {len(class_names)} Scripts")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    # Create legend with class names (show subset if too many)
    if len(class_names) <= 20:
        handles = [plt.scatter([], [], c=[cmap(i)], label=name)
                   for i, name in enumerate(class_names)]
        plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5),
                   fontsize=8, ncol=2)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()

    plt.close()


def traverse_latent_dimension(
    model,
    config,
    device: torch.device,
    dimension: int,
    steps: int = 11,
    range_std: float = 3.0,
    base_latent: Optional[torch.Tensor] = None,
    output_path: Optional[Path] = None,
):
    """Traverse a single latent dimension while holding others fixed."""
    latent_dim = config.latent_dim

    if dimension >= latent_dim:
        raise ValueError(f"Dimension {dimension} out of range (max: {latent_dim - 1})")

    # Use provided base or zeros
    if base_latent is None:
        base_latent = torch.zeros(1, latent_dim, device=device)
    else:
        base_latent = base_latent.clone()

    # Generate range of values
    values = torch.linspace(-range_std, range_std, steps)

    images = []
    with torch.no_grad():
        for val in values:
            z = base_latent.clone()
            z[0, dimension] = val
            img = model.decode(z)
            images.append(img)

    # Combine into grid
    images = torch.cat(images, dim=0)

    if output_path:
        save_image(images, output_path, nrow=steps, normalize=False)
        print(f"Saved traversal to: {output_path}")
    else:
        # Display
        grid = make_grid(images, nrow=steps, normalize=False)
        plt.figure(figsize=(steps * 1.5, 2))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        plt.title(f"Latent Dimension {dimension} Traversal")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close()

    return images


def traverse_all_dimensions(
    model,
    config,
    device: torch.device,
    output_dir: Path,
    steps: int = 11,
    range_std: float = 3.0,
    base_image_path: Optional[Path] = None,
):
    """Traverse all latent dimensions and save visualizations."""
    output_dir.mkdir(exist_ok=True)

    # Get base latent if image provided
    if base_image_path:
        transform = get_transform()
        image = load_image(base_image_path, transform, device)
        with torch.no_grad():
            base_latent = model.get_latent(image)
        print(f"Using base image: {base_image_path}")
    else:
        base_latent = None

    print(f"Traversing {config.latent_dim} dimensions...")

    all_traversals = []
    for dim in tqdm(range(config.latent_dim)):
        images = traverse_latent_dimension(
            model, config, device, dim, steps, range_std, base_latent,
            output_path=output_dir / f"dim_{dim:03d}.png"
        )
        all_traversals.append(images)

    # Create summary grid (subset of dimensions)
    num_dims_to_show = min(16, config.latent_dim)
    summary_images = []
    for i in range(0, config.latent_dim, max(1, config.latent_dim // num_dims_to_show)):
        summary_images.append(all_traversals[i])

    summary = torch.cat(summary_images[:num_dims_to_show], dim=0)
    save_image(
        summary,
        output_dir / "summary_traversal.png",
        nrow=steps,
        normalize=False
    )
    print(f"Saved summary to: {output_dir / 'summary_traversal.png'}")


def interpolate_characters(
    model,
    config,
    device: torch.device,
    image1_path: Path,
    image2_path: Path,
    steps: int = 11,
    output_path: Optional[Path] = None,
):
    """Interpolate between two characters in latent space."""
    transform = get_transform()

    # Load images
    img1 = load_image(image1_path, transform, device)
    img2 = load_image(image2_path, transform, device)

    # Get latent representations
    with torch.no_grad():
        z1 = model.get_latent(img1)
        z2 = model.get_latent(img2)

    # Interpolate
    images = [img1]
    with torch.no_grad():
        for i in range(1, steps - 1):
            alpha = i / (steps - 1)
            z = (1 - alpha) * z1 + alpha * z2
            img = model.decode(z)
            images.append(img)
    images.append(img2)

    # Combine
    images = torch.cat(images, dim=0)

    if output_path:
        save_image(images, output_path, nrow=steps, normalize=False)
        print(f"Saved interpolation to: {output_path}")
    else:
        grid = make_grid(images, nrow=steps, normalize=False)
        plt.figure(figsize=(steps * 1.5, 2))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        plt.title(f"Interpolation: {image1_path.stem} → {image2_path.stem}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close()

    return images


def find_nearest_neighbors(
    model,
    config,
    data_dir: Path,
    device: torch.device,
    query_path: Path,
    k: int = 10,
    output_path: Optional[Path] = None,
):
    """Find k nearest neighbors in latent space."""
    # Get query latent
    transform = get_transform()
    query_img = load_image(query_path, transform, device)
    with torch.no_grad():
        query_z = model.get_latent(query_img).cpu().numpy().squeeze()

    # Collect all latents
    latents, labels, class_names, paths = collect_latents(model, data_dir, device)

    # Compute distances
    distances = np.linalg.norm(latents - query_z, axis=1)
    nearest_indices = np.argsort(distances)[:k + 1]  # +1 to include query

    print(f"\nNearest neighbors for: {query_path}")
    print("-" * 50)

    images = [query_img]
    for idx in nearest_indices:
        if paths[idx] != query_path:
            print(f"  {distances[idx]:.4f}: {paths[idx]} ({class_names[labels[idx]]})")
            img = load_image(paths[idx], transform, device)
            images.append(img)
            if len(images) > k:
                break

    # Visualize
    images = torch.cat(images, dim=0)

    if output_path:
        save_image(images, output_path, nrow=min(k + 1, 11), normalize=False)
        print(f"\nSaved neighbors to: {output_path}")
    else:
        grid = make_grid(images, nrow=min(k + 1, 11), normalize=False)
        plt.figure(figsize=(min(k + 1, 11) * 1.5, 2))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        plt.title(f"Query and {k} Nearest Neighbors")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close()


def sample_from_latent(
    model,
    config,
    device: torch.device,
    num_samples: int = 64,
    output_path: Optional[Path] = None,
):
    """Generate random samples from the latent space."""
    with torch.no_grad():
        samples = model.sample(num_samples, device)

    nrow = int(np.sqrt(num_samples))

    if output_path:
        save_image(samples, output_path, nrow=nrow, normalize=False)
        print(f"Saved {num_samples} samples to: {output_path}")
    else:
        grid = make_grid(samples, nrow=nrow, normalize=False)
        plt.figure(figsize=(nrow * 1.5, nrow * 1.5))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        plt.title(f"Random Samples from Latent Space")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close()


def analyze_latent_statistics(
    model,
    config,
    data_dir: Path,
    device: torch.device,
    output_dir: Optional[Path] = None,
):
    """Analyze statistics of the latent space."""
    latents, labels, class_names, _ = collect_latents(model, data_dir, device)

    print("\n" + "=" * 60)
    print("Latent Space Statistics")
    print("=" * 60)

    # Global statistics
    print(f"\nGlobal Statistics:")
    print(f"  Samples: {len(latents)}")
    print(f"  Dimensions: {latents.shape[1]}")
    print(f"  Mean: {latents.mean():.4f}")
    print(f"  Std: {latents.std():.4f}")
    print(f"  Min: {latents.min():.4f}")
    print(f"  Max: {latents.max():.4f}")

    # Per-dimension statistics
    dim_means = latents.mean(axis=0)
    dim_stds = latents.std(axis=0)

    print(f"\nPer-Dimension Statistics:")
    print(f"  Mean range: [{dim_means.min():.4f}, {dim_means.max():.4f}]")
    print(f"  Std range: [{dim_stds.min():.4f}, {dim_stds.max():.4f}]")

    # Most/least variable dimensions
    sorted_by_std = np.argsort(dim_stds)
    print(f"\n  Most variable dimensions: {sorted_by_std[-5:][::-1].tolist()}")
    print(f"  Least variable dimensions: {sorted_by_std[:5].tolist()}")

    # Per-class analysis
    print(f"\nPer-Class Latent Means (showing first 10 classes):")
    labels_array = np.array(labels)
    for i, class_name in enumerate(class_names[:10]):
        class_mask = labels_array == i
        class_latents = latents[class_mask]
        if len(class_latents) > 0:
            print(f"  {class_name:20s}: mean={class_latents.mean():.4f}, std={class_latents.std():.4f}")

    if output_dir:
        output_dir.mkdir(exist_ok=True)

        # Save dimension statistics plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].bar(range(len(dim_means)), dim_means)
        axes[0].set_xlabel("Latent Dimension")
        axes[0].set_ylabel("Mean")
        axes[0].set_title("Mean per Latent Dimension")

        axes[1].bar(range(len(dim_stds)), dim_stds)
        axes[1].set_xlabel("Latent Dimension")
        axes[1].set_ylabel("Std")
        axes[1].set_title("Standard Deviation per Latent Dimension")

        plt.tight_layout()
        plt.savefig(output_dir / "latent_statistics.png", dpi=150)
        print(f"\nSaved statistics plot to: {output_dir / 'latent_statistics.png'}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Explore the latent space of a trained VAE",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints_vae/best_vae.pt"),
        help="Path to VAE checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("output"),
        help="Directory containing character images",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Visualize latent space")
    viz_parser.add_argument("--method", choices=["tsne", "umap", "pca"], default="tsne")
    viz_parser.add_argument("--output", type=Path, help="Output path for image")
    viz_parser.add_argument("--max-per-class", type=int, default=100)

    # Traverse command
    trav_parser = subparsers.add_parser("traverse", help="Traverse latent dimensions")
    trav_parser.add_argument("--dim", type=int, help="Specific dimension to traverse (or all if not set)")
    trav_parser.add_argument("--steps", type=int, default=11)
    trav_parser.add_argument("--range", type=float, default=3.0)
    trav_parser.add_argument("--base-image", type=Path, help="Base image for traversal")
    trav_parser.add_argument("--output", type=Path, help="Output path")

    # Interpolate command
    interp_parser = subparsers.add_parser("interpolate", help="Interpolate between characters")
    interp_parser.add_argument("--char1", type=Path, required=True)
    interp_parser.add_argument("--char2", type=Path, required=True)
    interp_parser.add_argument("--steps", type=int, default=11)
    interp_parser.add_argument("--output", type=Path)

    # Neighbors command
    neigh_parser = subparsers.add_parser("neighbors", help="Find nearest neighbors")
    neigh_parser.add_argument("--char", type=Path, required=True)
    neigh_parser.add_argument("--k", type=int, default=10)
    neigh_parser.add_argument("--output", type=Path)

    # Sample command
    sample_parser = subparsers.add_parser("sample", help="Sample from latent space")
    sample_parser.add_argument("--num", type=int, default=64)
    sample_parser.add_argument("--output", type=Path)

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Analyze latent statistics")
    stats_parser.add_argument("--output-dir", type=Path)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Load model
    device = get_device(args.device)
    print(f"Using device: {device}")
    print(f"Loading model from: {args.checkpoint}")

    model, config, idx_to_class = load_model(args.checkpoint, device)
    print(f"Model loaded (latent_dim={config.latent_dim})")

    # Execute command
    if args.command == "visualize":
        visualize_latent_space(
            model, config, idx_to_class, args.data_dir, device,
            method=args.method,
            output_path=args.output,
            max_per_class=args.max_per_class,
        )

    elif args.command == "traverse":
        if args.dim is not None:
            traverse_latent_dimension(
                model, config, device,
                dimension=args.dim,
                steps=args.steps,
                range_std=args.range,
                output_path=args.output,
            )
        else:
            output_dir = args.output or Path("latent_traversals")
            traverse_all_dimensions(
                model, config, device, output_dir,
                steps=args.steps,
                range_std=args.range,
                base_image_path=args.base_image,
            )

    elif args.command == "interpolate":
        interpolate_characters(
            model, config, device,
            args.char1, args.char2,
            steps=args.steps,
            output_path=args.output,
        )

    elif args.command == "neighbors":
        find_nearest_neighbors(
            model, config, args.data_dir, device,
            query_path=args.char,
            k=args.k,
            output_path=args.output,
        )

    elif args.command == "sample":
        sample_from_latent(
            model, config, device,
            num_samples=args.num,
            output_path=args.output,
        )

    elif args.command == "stats":
        analyze_latent_statistics(
            model, config, args.data_dir, device,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
