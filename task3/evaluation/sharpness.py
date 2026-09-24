"""Local sharpness proxy: cross-entropy change under normalized ascent perturbation."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn


def sharpness_proxy(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    radius: float = 0.05,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    x, y = x.to(device), y.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    backup = [p.detach().clone() for p in params]

    loss0 = F.cross_entropy(model(x), y)
    grads = torch.autograd.grad(loss0, params, allow_unused=True)
    grad_norm = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads if g is not None))
    scale = radius / (grad_norm + 1e-12)

    with torch.no_grad():
        for p, g in zip(params, grads):
            if g is not None:
                p.add_(g * scale)
        loss1 = F.cross_entropy(model(x), y)
        for p, b in zip(params, backup):
            p.copy_(b)

    if was_training:
        model.train()
    return {
        "loss": float(loss0.item()),
        "perturbed_loss": float(loss1.item()),
        "delta_sharp": float(loss1.item() - loss0.item()),
        "grad_norm": float(grad_norm.item()),
        "radius": float(radius),
        "n": int(x.size(0)),
    }