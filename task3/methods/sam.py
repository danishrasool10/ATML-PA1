"""Non-adaptive Sharpness-Aware Minimization using AdamW."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping

import torch
from torch import nn

from task3.methods.erm import balanced_ce


class SAMMethod:
    name = "sam"

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, cfg: Mapping[str, Any]) -> None:
        self.model = model
        self.optimizer = optimizer
        self.rho = float(cfg["rho"])
        self.params = [p for p in model.parameters() if p.requires_grad]

    def train_step(self, x: torch.Tensor, y: torch.Tensor, sizes: List[int]) -> Dict[str, torch.Tensor]:
        # Gradient of source loss at theta
        self.optimizer.zero_grad(set_to_none=True)
        loss = balanced_ce(self.model(x), y, sizes)
        loss.backward()

        # Compute normalized perturbation and step to theta + eps
        grads = [p.grad for p in self.params if p.grad is not None]
        grad_norm = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))
        scale = self.rho / (grad_norm + 1e-12)
        backup = [p.detach().clone() for p in self.params]
        with torch.no_grad():
            for p in self.params:
                if p.grad is not None:
                    p.add_(p.grad * scale)

        # Gradient at theta + eps
        self.optimizer.zero_grad(set_to_none=True)
        loss_perturbed = balanced_ce(self.model(x), y, sizes)
        loss_perturbed.backward()

        # Restore theta, then apply optimizer step using perturbed gradient
        with torch.no_grad():
            for p, b in zip(self.params, backup):
                p.copy_(b)
        self.optimizer.step()
        return {
            "loss": loss.detach(),
            "cls_loss": loss.detach(),
            "perturbed_loss": loss_perturbed.detach(),
            "grad_norm": grad_norm.detach(),
        }