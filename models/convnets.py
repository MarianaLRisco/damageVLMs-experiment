"""ConvNet model builders for image-only baselines."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torchvision.models as models


def _resolve_pretrained_weight(weights):
    """Return default pretrained weights if available, else None."""
    if hasattr(weights, "DEFAULT"):
        return weights.DEFAULT
    return None


def _build_backbone(model_name: str, num_classes: int, use_pretrained: bool):
    """Build the requested torchvision backbone with a replaced classifier head."""
    train_layers: List[nn.Module] = []

    if model_name == "resnet50":
        weights = _resolve_pretrained_weight(models.ResNet50_Weights) if use_pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        train_layers = list(model.children())[:-1]

    elif model_name == "efficientnet_b0":
        weights = _resolve_pretrained_weight(models.EfficientNet_B0_Weights) if use_pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        train_layers = list(model.features)

    elif model_name == "vgg16":
        weights = _resolve_pretrained_weight(models.VGG16_Weights) if use_pretrained else None
        model = models.vgg16(weights=weights)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        train_layers = list(model.features)

    else:
        raise ValueError(f"Model '{model_name}' not supported")

    return model, train_layers


def build_model(
    model_name: str,
    num_classes: int,
    trainable_layers: int = 2,
    use_pretrained: bool = True,
) -> nn.Module:
    """Build and freeze a ConvNet backbone.

    By default, only a few backbone layers and the final classifier are unfrozen,
    so training is lightweight and stable for local experiments.
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if trainable_layers < 0:
        raise ValueError("trainable_layers must be >= 0")

    model, trainable_blocks = _build_backbone(model_name, num_classes, use_pretrained)

    for param in model.parameters():
        param.requires_grad = False

    for layer in trainable_blocks[-max(trainable_layers, 0):]:
        for param in layer.parameters():
            param.requires_grad = True

    if hasattr(model, "fc"):
        for param in model.fc.parameters():
            param.requires_grad = True
    if hasattr(model, "classifier"):
        # For VGG and EfficientNet, keep only the final head trainable.
        for param in model.classifier.parameters():
            param.requires_grad = True

    return model
