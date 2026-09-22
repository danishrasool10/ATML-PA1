"""Gradient-reversal layer, GRL schedule and the domain discriminator shared by DANN and CDAN."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return GradientReversalFunction.apply(x, alpha)


def grl_schedule(progress: float, gamma: float = 10.0, max_value: float = 1.0) -> float:
    """alpha(p) = max_value * (2 / (1 + exp(-gamma * p)) - 1), p in [0, 1]."""
    return max_value * (2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0)


class DomainDiscriminator(nn.Module):
    """[GRL] -> Linear(in, 256) -> ReLU -> Dropout(0.5) -> Linear(256, 2)."""

    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return self.net(grad_reverse(x, alpha))