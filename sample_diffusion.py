#!/usr/bin/env python3
"""
Generate new Unicode character images using a trained diffusion model.

Usage:
    # Generate random samples
    python sample_diffusion.py --num 64 --output samples.png

    # Generate samples for specific scripts
    python sample_diffusion.py --class latin --num 16
    python sample_diffusion.py --class-idx 0 1 2 3 --num 4

    # Use DDIM for faster sampling
    python sample_diffusion.py --ddim --ddim-steps 50

    # Adjust guidance strength
    python sample_diffusion.py --cfg-scale 5.0 --class hiragana
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).parent))

from diffusion.model import SimpleUNet
from diffusion.diffusion import GaussianDiffusion
from diffusion.trainer import get_device


def load_model(checkpoint_path: Path, device: torch.device):
    """Load a trained diffusion model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    idx_to_class = checkpoint.get("idx_to_class", {})

    # Create model
    model = SimpleUNet(
        image_channels=config.image_channels,
        base_channels=config.base_channels,
        channel_mults=config.channel_mults,
        dropout=0.0,  # No dropout during inference
        num_classes=config.num_classes if config.conditional else None,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Create diffusion
    diffusion = GaussianDiffusion(
        model=model,
        image_size=config.image_size,
        image_channels=config.image_channels,
        timesteps=config.num_timesteps,
        beta_schedule=config.beta_schedule,
    )
    diffusion = diffusion.to(device)

    return diffusion, config, idx_to_class


def main():
    parser = argparse.ArgumentParser(
        description="Generate Unicode characters with a trained diffusion model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints_diffusion/best_diffusion.pt"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=64,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: prints to screen info)",
    )
    parser.add_argument(
        "--class",
        type=str,
        dest="class_name",
        default=None,
        help="Generate samples for a specific script (e.g., 'latin', 'hiragana')",
    )
    parser.add_argument(
        "--class-idx",
        type=int,
        nargs="+",
        default=None,
        help="Generate samples for specific class indices",
    )
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Generate one sample per class",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale (higher = more class-specific)",
    )
    parser.add_argument(
        "--ddim",
        action="store_true",
        help="Use DDIM sampling (faster)",
    )
    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=50,
        help="Number of DDIM steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--list-classes",
        action="store_true",
        help="List available classes and exit",
    )

    args = parser.parse_args()

    # Load model
    device = get_device(args.device)
    print(f"Using device: {device}")
    print(f"Loading model from: {args.checkpoint}")

    diffusion, config, idx_to_class = load_model(args.checkpoint, device)

    # Create reverse mapping
    class_to_idx = {v: int(k) for k, v in idx_to_class.items()}

    if args.list_classes:
        print("\nAvailable classes:")
        for idx in sorted(idx_to_class.keys(), key=int):
            print(f"  {idx}: {idx_to_class[idx]}")
        return

    print(f"Model loaded: {config.num_classes} classes, {config.num_timesteps} timesteps")

    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Determine class labels
    class_labels = None
    num_samples = args.num

    if args.all_classes:
        # One sample per class
        num_samples = config.num_classes
        class_labels = torch.arange(num_samples, device=device)
        print(f"Generating one sample per class ({num_samples} total)")

    elif args.class_name is not None:
        # Specific class by name
        if args.class_name not in class_to_idx:
            print(f"Error: Unknown class '{args.class_name}'")
            print(f"Available classes: {list(class_to_idx.keys())}")
            return
        class_idx = class_to_idx[args.class_name]
        class_labels = torch.full((num_samples,), class_idx, device=device, dtype=torch.long)
        print(f"Generating {num_samples} samples of class '{args.class_name}' (idx={class_idx})")

    elif args.class_idx is not None:
        # Specific class indices
        class_labels = []
        samples_per_class = num_samples // len(args.class_idx)
        for idx in args.class_idx:
            class_labels.extend([idx] * samples_per_class)
        class_labels = torch.tensor(class_labels[:num_samples], device=device, dtype=torch.long)
        num_samples = len(class_labels)
        print(f"Generating {num_samples} samples across classes {args.class_idx}")

    else:
        # Unconditional
        print(f"Generating {num_samples} unconditional samples")

    # Generate samples
    print(f"\nSampling with {'DDIM' if args.ddim else 'DDPM'}, CFG scale={args.cfg_scale}...")

    with torch.no_grad():
        if args.ddim:
            samples = diffusion.sample_ddim(
                num_samples,
                class_labels=class_labels,
                cfg_scale=args.cfg_scale,
                ddim_steps=args.ddim_steps,
                device=device,
                show_progress=True,
            )
        else:
            samples = diffusion.sample(
                num_samples,
                class_labels=class_labels,
                cfg_scale=args.cfg_scale,
                device=device,
                show_progress=True,
            )

    # Rescale from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)

    # Save or display
    if args.output:
        nrow = min(8, int(num_samples ** 0.5))
        save_image(samples, args.output, nrow=nrow, normalize=False)
        print(f"\nSaved {num_samples} samples to: {args.output}")
    else:
        # Default output path
        output_path = Path("generated_samples.png")
        nrow = min(8, int(num_samples ** 0.5))
        save_image(samples, output_path, nrow=nrow, normalize=False)
        print(f"\nSaved {num_samples} samples to: {output_path}")

    # Print class labels if conditional
    if class_labels is not None and len(torch.unique(class_labels)) <= 10:
        print("\nGenerated classes:")
        for i, label in enumerate(class_labels[:min(16, len(class_labels))]):
            class_name = idx_to_class.get(str(label.item()), f"class_{label.item()}")
            print(f"  Sample {i}: {class_name}")
        if len(class_labels) > 16:
            print(f"  ... and {len(class_labels) - 16} more")


if __name__ == "__main__":
    main()
