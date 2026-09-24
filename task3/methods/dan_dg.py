"""DAN-DG: ERM with pairwise multi-kernel MMD alignment across source domains."""
from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Mapping, Sequence

import torch
from torch import nn

from task3.methods.erm import balanced_ce


def multi_kernel_mmd2(x: torch.Tensor, y: torch.Tensor, multipliers: Sequence[float] = (0.5, 1.0, 2.0)) -> torch.Tensor:
    """Biased V-statistic MMD^2 with a sum of RBF kernels scaled by median pairwise squared distance."""
    n = x.size(0)
    z = torch.cat([x, y], dim=0)
    sq = z.pow(2).sum(dim=1)
    d2 = (sq[:, None] + sq[None, :] - 2.0 * (z @ z.t())).clamp_min(0.0)
    off_diag = ~torch.eye(z.size(0), dtype=torch.bool, device=z.device)
    med = d2.detach()[off_diag].median().clamp_min(1e-8)
    k = sum(torch.exp(-d2 / (m * med)) for m in multipliers)
    k_xx, k_yy, k_xy = k[:n, :n], k[n:, n:], k[:n, n:]
    return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()


def pairwise_source_mmd(feats: Sequence[torch.Tensor], multipliers: Sequence[float]) -> torch.Tensor:
    pairs = list(combinations(range(len(feats)), 2))
    total = sum(multi_kernel_mmd2(feats[i], feats[j], multipliers) for i, j in pairs)
    return total / len(pairs)


class DANDGMethod:
    name = "dan_dg"

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, cfg: Mapping[str, Any]) -> None:
        self.model = model
        self.optimizer = optimizer
        self.lambda_dg = float(cfg["lambda_dg"])
        self.multipliers = [float(m) for m in cfg.get("mmd_bandwidth_multipliers", [0.5, 1.0, 2.0])]

    def train_step(self, x: torch.Tensor, y: torch.Tensor, sizes: List[int]) -> Dict[str, torch.Tensor]:
        logits, feats = self.model(x, return_features=True)
        cls_loss = balanced_ce(logits, y, sizes)
        mmd = pairwise_source_mmd(torch.split(feats, sizes), self.multipliers)
        loss = cls_loss + self.lambda_dg * mmd
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.detach(), "cls_loss": cls_loss.detach(), "mmd": mmd.detach()}