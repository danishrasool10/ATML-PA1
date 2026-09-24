"""Linear classifier head and full model."""
from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import nn

from task3.models.backbone import FEATURE_DIM, ResNet18Backbone


class ClassifierHead(nn.Module):
    def __init__(self, in_features: int = FEATURE_DIM, num_classes: int = 7) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


class DGModel(nn.Module):
    def __init__(self, num_classes: int = 7, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = ResNet18Backbone(pretrained=pretrained)
        self.head = ClassifierHead(FEATURE_DIM, num_classes)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        features = self.backbone(x)
        logits = self.head(features)
        if return_features:
            return logits, features
        return logits


def load_dg_checkpoint(path: str, num_classes: int, device: torch.device) -> DGModel:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    model = DGModel(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()