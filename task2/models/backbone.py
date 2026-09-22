"""ResNet-18 feature extractor and frozen-BatchNorm policy."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

_BatchNorm = nn.modules.batchnorm._BatchNorm


class ResNet18Backbone(nn.Module):
    feature_dim = 512

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)
        net.fc = nn.Identity()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def freeze_batchnorm_statistics(module: nn.Module) -> None:
    """Call after module.train() to put BatchNorm layers in eval mode."""
    for m in module.modules():
        if isinstance(m, _BatchNorm):
            m.eval()


def snapshot_bn_buffers(module: nn.Module) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    return {
        name: (m.running_mean.detach().clone(), m.running_var.detach().clone())
        for name, m in module.named_modules()
        if isinstance(m, _BatchNorm)
    }


def verify_bn_buffers_unchanged(module: nn.Module, snapshot: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> None:
    for name, m in module.named_modules():
        if name in snapshot:
            rm, rv = snapshot[name]
            if not (torch.equal(m.running_mean, rm) and torch.equal(m.running_var, rv)):
                raise RuntimeError(f"BatchNorm running statistics of '{name}' changed during training.")