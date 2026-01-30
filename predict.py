#!/usr/bin/env python3
"""
Predict the script/language of Unicode character images.

Usage:
    python predict.py image.png [image2.png ...]
    python predict.py --checkpoint path/to/model.pt image.png
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))

from ml.model import create_model, get_device


def load_model(checkpoint_path: Path, device: torch.device) -> tuple:
    """Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Device to load the model on.

    Returns:
        Tuple of (model, idx_to_class, config).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint["config"]
    idx_to_class = checkpoint["idx_to_class"]

    model = create_model(
        num_classes=config.num_classes,
        model_name=config.model_name,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, idx_to_class, config


def get_inference_transform() -> transforms.Compose:
    """Get transforms for inference."""
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    image_path: Path,
    transform: transforms.Compose,
    idx_to_class: dict,
    device: torch.device,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Predict the class of an image.

    Args:
        model: Trained model.
        image_path: Path to the image file.
        transform: Image transforms.
        idx_to_class: Mapping from indices to class names.
        device: Device to use.
        top_k: Number of top predictions to return.

    Returns:
        List of (class_name, probability) tuples.
    """
    # Load and transform image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Get predictions
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)[0]

    # Get top-k predictions
    top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(idx_to_class)))

    results = []
    for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
        class_name = idx_to_class.get(int(idx), f"class_{idx}")
        results.append((class_name, float(prob)))

    return results


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(
        description="Predict Unicode character script/language",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "images",
        type=Path,
        nargs="+",
        help="Image file(s) to classify",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best_model.pt"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to show",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use",
    )

    args = parser.parse_args()

    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model, idx_to_class, config = load_model(args.checkpoint, device)
    print(f"Model loaded: {config.model_name} with {config.num_classes} classes")

    # Get transforms
    transform = get_inference_transform()

    # Process each image
    print("\n" + "=" * 60)
    print("Predictions")
    print("=" * 60)

    for image_path in args.images:
        if not image_path.exists():
            print(f"\nError: File not found: {image_path}")
            continue

        print(f"\n{image_path}:")
        predictions = predict_image(
            model, image_path, transform, idx_to_class, device, args.top_k
        )

        for i, (class_name, prob) in enumerate(predictions, 1):
            bar = "█" * int(prob * 30)
            print(f"  {i}. {class_name:25s} {prob:6.2%} {bar}")


if __name__ == "__main__":
    main()
