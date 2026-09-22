"""Linear head and full ResNet-18 classifier."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from task2.models.backbone import ResNet18Backbone


class ClassifierHead(nn.Module):
    def __init__(self, in_features: int = 512, num_classes: int = 7):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


class UDAClassifier(nn.Module):
    def __init__(self, backbone: ResNet18Backbone, head: ClassifierHead):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.feature_dim = backbone.feature_dim
        self.num_classes = head.fc.out_features

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        return self.head(feats), feats


def build_classifier(num_classes: int = 7, pretrained: bool = True) -> UDAClassifier:
    backbone = ResNet18Backbone(pretrained=pretrained)
    head = ClassifierHead(backbone.feature_dim, num_classes)
    return UDAClassifier(backbone, head)