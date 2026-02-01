"""Image transforms for different model types."""

from torchvision import transforms


# ImageNet normalization (for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Identity:
    """Identity transform that returns input unchanged. Picklable alternative to Lambda."""
    def __call__(self, x):
        return x


def get_normalize_transform(mode: str = "classification"):
    """Get normalization transform based on mode.

    Args:
        mode: One of "classification", "vae", "diffusion".

    Returns:
        Normalization transform.
    """
    if mode == "classification":
        # ImageNet normalization for pretrained ResNet
        return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    elif mode == "vae":
        # No normalization - keep in [0, 1] for BCE loss
        return Identity()
    elif mode == "diffusion":
        # Scale to [-1, 1] for diffusion
        return transforms.Normalize(mean=[0.5], std=[0.5])
    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_transforms(
    mode: str = "classification",
    train: bool = True,
    image_size: int = 64,
) -> transforms.Compose:
    """Get image transforms for training or evaluation.

    Args:
        mode: One of "classification", "vae", "diffusion".
        train: Whether to include training augmentations.
        image_size: Target image size.

    Returns:
        Composed transforms.
    """
    transform_list = []

    # Resize if needed
    if image_size != 64:
        transform_list.append(transforms.Resize((image_size, image_size)))

    # Training augmentations
    if train:
        if mode == "classification":
            transform_list.extend([
                transforms.RandomRotation(10),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                ),
                transforms.RandomInvert(p=0.1),
            ])
        elif mode == "vae":
            # Light augmentation for VAE
            transform_list.extend([
                transforms.RandomAffine(
                    degrees=5,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                ),
            ])
        elif mode == "diffusion":
            # Moderate augmentation for diffusion
            transform_list.extend([
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                ),
            ])

    # Convert to tensor
    transform_list.append(transforms.ToTensor())

    # Apply normalization
    transform_list.append(get_normalize_transform(mode))

    return transforms.Compose(transform_list)
