"""CIFAR ResNet-18 with 3x3 stride-1 stem and no max-pool."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18Cifar(nn.Module):
    """num_classes=10 for Vanilla/GCSC; num_classes=15 for PROSER."""

    feature_dim = 512

    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self._in_planes = 64
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, planes, blocks, stride):
        layers = []
        for s in [stride] + [1] * (blocks - 1):
            layers.append(BasicBlock(self._in_planes, planes, s))
            self._in_planes = planes
        return nn.Sequential(*layers)

    def forward_pre(self, x):
        """Network up to and including layer2."""
        out = F.relu(self.bn1(self.conv1(x)))
        return self.layer2(self.layer1(out))

    def forward_post(self, h):
        """layer3 -> layer4 -> global average pool; returns the 512-d penultimate feature."""
        h = self.layer4(self.layer3(h))
        return torch.flatten(self.avgpool(h), 1)

    def forward_from_pre(self, h, return_features=False):
        feat = self.forward_post(h)
        logits = self.fc(feat)
        return (logits, feat) if return_features else logits

    def forward(self, x, return_features=False):
        return self.forward_from_pre(self.forward_pre(x), return_features)