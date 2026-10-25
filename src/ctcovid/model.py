"""Transfer-learning model: ImageNet MobileNetV3 with the final layer swapped.

Video: 'Transfer Learning' slide (ImageNet -> medical dataset) then the
MobileNetV3 diagram ("swapped" the head for our classes).
"""
from __future__ import annotations

import warnings

from torch import nn
from torchvision import models

_BUILDERS = {
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V1),
    "mobilenet_v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
}


def build_model(arch: str = "mobilenet_v3_large", num_classes: int = 3,
                pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    builder, weights = _BUILDERS[arch]
    try:
        model = builder(weights=weights if pretrained else None)
    except Exception as e:  # noqa: BLE001 - offline / blocked download
        warnings.warn(f"Could not load pretrained weights ({e}); using random init.")
        model = builder(weights=None)


    in_f = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, num_classes)  # the "swap"

    set_backbone_trainable(model, trainable=not freeze_backbone)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for p in model.features.parameters():
        p.requires_grad = trainable
    for p in model.classifier.parameters():
        p.requires_grad = True


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train


def get_cam_target_layer(model: nn.Module) -> nn.Module:
    """Return the final feature extraction layer to use as Grad-CAM target."""
    return model.features[-1]

