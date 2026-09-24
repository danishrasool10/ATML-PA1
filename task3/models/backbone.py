"""ResNet-18 backbone, transforms, and frozen BatchNorm utilities."""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FEATURE_DIM = 512


class ResNet18Backbone(nn.Module):
    """ResNet-18 feature extractor outputting 512-d pooled representations."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)
        net.fc = nn.Identity()
        self.net = net
        self.feature_dim = FEATURE_DIM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_transforms(resize: int = 256, crop: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_tf = transforms.Compose(
        [
            transforms.Resize((resize, resize)),
            transforms.RandomCrop(crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((resize, resize)),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tf, eval_tf


def train_with_frozen_bn(model: nn.Module) -> nn.Module:
    """Put model in train mode while keeping BatchNorm in eval mode to freeze running stats."""
    model.train()
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            m.eval()
    return model


def assert_bn_frozen(model: nn.Module) -> None:
    for name, m in model.named_modules():
        if isinstance(m, _BatchNorm) and m.training:
            raise RuntimeError(f"BatchNorm module '{name}' is in training mode; running statistics would be updated.")


def bn_fingerprint(model: nn.Module) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            out.append(m.running_mean.detach().clone().cpu())
            out.append(m.running_var.detach().clone().cpu())
    return out


def bn_unchanged(before: List[torch.Tensor], after: List[torch.Tensor]) -> bool:
    return len(before) == len(after) and all(torch.equal(a, b) for a, b in zip(before, after))