"""ResNet classifier for script/language classification."""

import torch.nn as nn
from torchvision import models


def create_classifier(
    num_classes: int,
    model_name: str = "resnet18",
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Create a ResNet classifier for character classification.

    Args:
        num_classes: Number of output classes.
        model_name: Model architecture ("resnet18", "resnet34", "resnet50").
        pretrained: Whether to use pretrained ImageNet weights.
        dropout: Dropout rate before final layer.

    Returns:
        PyTorch model.
    """
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
    elif model_name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        in_features = model.fc.in_features
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Replace classifier head
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )

    return model
