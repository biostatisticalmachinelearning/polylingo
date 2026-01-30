#!/usr/bin/env python3
"""
Sample from a trained diffusion model.

Usage:
    python scripts/sample_diffusion.py [--checkpoint PATH] [--num-samples N]

Generate new Unicode character images using a trained diffusion model.
"""

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from polylingo import (
    DiffusionConfig,
    SimpleUNet,
    GaussianDiffusion,
    get_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample from a trained diffusion model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/diffusion/best_diffusion.pt"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--num-samples", type=int, default=64,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("samples"),
        help="Directory to save generated samples",
    )
    parser.add_argument(
        "--cfg-scale", type=float, default=3.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--use-ddim", action="store_true", default=True,
        help="Use DDIM sampling (faster)",
    )
    parser.add_argument(
        "--ddim-steps", type=int, default=50,
        help="Number of DDIM steps",
    )
    parser.add_argument(
        "--class-label", type=int, default=None,
        help="Generate samples for a specific class (None for all classes)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    device = get_device(args.device)
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    config = checkpoint["config"]
    idx_to_class = checkpoint.get("idx_to_class", {})

    print(f"Model trained on {len(idx_to_class)} classes")

    # Recreate model
    model = SimpleUNet(
        image_channels=config.image_channels,
        base_channels=config.base_channels,
        channel_mults=config.channel_mults,
        dropout=config.dropout,
        num_classes=config.num_classes if config.conditional else None,
    )

    diffusion = GaussianDiffusion(
        model=model,
        image_size=config.image_size,
        image_channels=config.image_channels,
        timesteps=config.num_timesteps,
        beta_schedule=config.beta_schedule,
    )

    diffusion.model.load_state_dict(checkpoint["model_state_dict"])
    diffusion = diffusion.to(device)
    diffusion.eval()

    print(f"Generating {args.num_samples} samples...")

    # Determine class labels
    if args.class_label is not None:
        class_labels = torch.full((args.num_samples,), args.class_label, device=device)
        class_name = idx_to_class.get(args.class_label, str(args.class_label))
        print(f"Generating class: {class_name}")
    elif config.conditional:
        # Generate one sample per class, cycling through
        num_classes = len(idx_to_class)
        class_labels = torch.arange(args.num_samples, device=device) % num_classes
        print(f"Generating samples cycling through {num_classes} classes")
    else:
        class_labels = None
        print("Generating unconditional samples")

    # Generate samples
    with torch.no_grad():
        if args.use_ddim:
            samples = diffusion.sample_ddim(
                args.num_samples,
                class_labels,
                cfg_scale=args.cfg_scale if config.conditional else 1.0,
                ddim_steps=args.ddim_steps,
                device=device,
                show_progress=True,
            )
        else:
            samples = diffusion.sample(
                args.num_samples,
                class_labels,
                cfg_scale=args.cfg_scale if config.conditional else 1.0,
                device=device,
                show_progress=True,
            )

    # Rescale from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2

    # Save samples
    output_path = args.output_dir / "generated_samples.png"
    save_image(samples, output_path, nrow=8, normalize=False)
    print(f"Samples saved to {output_path}")

    # Save individual samples
    for i, sample in enumerate(samples):
        if class_labels is not None:
            label = class_labels[i].item()
            class_name = idx_to_class.get(label, str(label))
            filename = f"sample_{i:04d}_{class_name}.png"
        else:
            filename = f"sample_{i:04d}.png"
        save_image(sample, args.output_dir / filename, normalize=False)

    print(f"Individual samples saved to {args.output_dir}")


if __name__ == "__main__":
    main()
