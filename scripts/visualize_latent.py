#!/usr/bin/env python3
"""
Visualize VAE latent space using t-SNE.

Usage:
    python scripts/visualize_latent.py [--checkpoint PATH] [--output PATH]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

from polylingo import ConvVAE, UnicodeDataset, load_dataset_info, get_device
from polylingo.data.transforms import get_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize VAE latent space")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints_vae/best_vae.pt"),
        help="Path to VAE checkpoint",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/unicode_chars"),
        help="Path to dataset",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("latent_tsne.png"),
        help="Output image path",
    )
    parser.add_argument(
        "--perplexity", type=float, default=30.0,
        help="t-SNE perplexity parameter",
    )
    parser.add_argument(
        "--max-samples", type=int, default=5000,
        help="Maximum samples to use (for speed)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device to use",
    )
    return parser.parse_args()


def encode_dataset(model, dataloader, device, labels_list):
    """Encode all images and return latent vectors."""
    model.eval()
    all_latents = []

    with torch.no_grad():
        for images in tqdm(dataloader, desc="Encoding images"):
            # Handle both (image,) and (image, label) returns
            if isinstance(images, (list, tuple)):
                images = images[0]
            images = images.to(device)
            mu, _ = model.encode(images)
            all_latents.append(mu.cpu().numpy())

    return np.vstack(all_latents), np.array(labels_list)


def main():
    args = parse_args()

    # Load checkpoint
    device = get_device(args.device)
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    # Handle old checkpoint format that references 'vae' module
    import sys
    import polylingo.configs
    sys.modules['vae'] = type(sys)('vae')
    sys.modules['vae.config'] = polylingo.configs

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("config")
    idx_to_class = checkpoint.get("idx_to_class", {})

    # Create model
    model = ConvVAE(
        latent_dim=config.latent_dim if config else 64,
        image_channels=1,
        hidden_dims=(32, 64, 128, 256),
        image_size=64,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded VAE with latent_dim={model.latent_dim}")

    # Load dataset
    print(f"\nLoading dataset from {args.data_dir}")
    paths, labels, idx_to_class_data, _ = load_dataset_info(args.data_dir)

    # Use idx_to_class from checkpoint if available, otherwise from data
    if not idx_to_class:
        idx_to_class = idx_to_class_data

    # Subsample if needed
    if len(paths) > args.max_samples:
        print(f"Subsampling to {args.max_samples} images...")
        indices = np.random.choice(len(paths), args.max_samples, replace=False)
        paths = [paths[i] for i in indices]
        labels = [labels[i] for i in indices]

    print(f"Using {len(paths)} images across {len(set(labels))} classes")

    # Create dataset and dataloader - use reconstruction mode for grayscale
    transform = get_transforms(mode="vae", train=False)
    dataset = UnicodeDataset(paths, labels, transform, mode="reconstruction")
    # Use num_workers=0 to avoid pickle issues with transforms
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Encode all images
    print("\nEncoding images through VAE...")
    latents, labels_arr = encode_dataset(model, dataloader, device, labels)
    print(f"Latent shape: {latents.shape}")

    # Apply t-SNE
    print(f"\nApplying t-SNE (perplexity={args.perplexity})...")
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        random_state=42,
        max_iter=1000,
        verbose=1,
    )
    latents_2d = tsne.fit_transform(latents)
    print(f"t-SNE complete! Output shape: {latents_2d.shape}")

    # Get unique classes and assign colors
    unique_labels = sorted(set(labels_arr))
    num_classes = len(unique_labels)

    # Use a colormap with enough distinct colors
    if num_classes <= 20:
        cmap = plt.cm.get_cmap('tab20', num_classes)
    else:
        cmap = plt.cm.get_cmap('nipy_spectral', num_classes)

    # Create plot
    print("\nCreating visualization...")
    fig, ax = plt.subplots(figsize=(16, 14))

    # Plot each class
    for i, label in enumerate(unique_labels):
        mask = labels_arr == label
        class_name = idx_to_class.get(label, str(label))
        ax.scatter(
            latents_2d[mask, 0],
            latents_2d[mask, 1],
            c=[cmap(i)],
            label=class_name,
            alpha=0.6,
            s=20,
        )

    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title(f"VAE Latent Space (t-SNE, {len(paths)} samples, {num_classes} scripts)", fontsize=14)

    # Create legend - place outside plot if many classes
    if num_classes <= 20:
        ax.legend(loc='best', fontsize=8, markerscale=1.5)
    else:
        # Put legend outside
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        ax.legend(
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            fontsize=7,
            markerscale=1.5,
            ncol=2 if num_classes > 40 else 1,
        )

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {args.output}")

    # Also create a version with just the top N most common scripts
    print("\nCreating simplified view with top 15 scripts...")

    # Count samples per class
    class_counts = {}
    for l in labels_arr:
        class_counts[l] = class_counts.get(l, 0) + 1

    top_classes = sorted(class_counts.keys(), key=lambda x: class_counts[x], reverse=True)[:15]
    top_mask = np.isin(labels_arr, top_classes)

    fig2, ax2 = plt.subplots(figsize=(14, 12))

    cmap2 = plt.cm.get_cmap('tab20', 15)
    for i, label in enumerate(top_classes):
        mask = labels_arr == label
        class_name = idx_to_class.get(label, str(label))
        count = class_counts[label]
        ax2.scatter(
            latents_2d[mask, 0],
            latents_2d[mask, 1],
            c=[cmap2(i)],
            label=f"{class_name} ({count})",
            alpha=0.7,
            s=25,
        )

    ax2.set_xlabel("t-SNE 1", fontsize=12)
    ax2.set_ylabel("t-SNE 2", fontsize=12)
    ax2.set_title("VAE Latent Space - Top 15 Scripts by Sample Count", fontsize=14)
    ax2.legend(loc='best', fontsize=9, markerscale=1.5)

    output_top = args.output.with_stem(args.output.stem + "_top15")
    plt.tight_layout()
    plt.savefig(output_top, dpi=150, bbox_inches='tight')
    print(f"Saved simplified visualization to: {output_top}")

    plt.show()


if __name__ == "__main__":
    main()
