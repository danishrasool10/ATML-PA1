"""Source-only ERM base class."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class LossOutput:
    total: torch.Tensor
    logs: Dict[str, Optional[float]] = field(default_factory=dict)

def make_logs(cls, align=None, total=None, domain_acc=None, alpha=None) -> Dict[str, Optional[float]]:
    return {
        "cls": float(cls.detach()),
        "align": None if align is None else float(align.detach()),
        "total": float((cls if total is None else total).detach()),
        "domain_acc": domain_acc,
        "alpha": alpha,
    }

class SourceOnlyMethod(nn.Module):
    name = "source_only"
    uses_target = False

    def __init__(self, cfg: dict, model: nn.Module):
        super().__init__()
        self.cfg = dict(cfg)

    def compute_loss(self, model, src_x, src_y, tgt_x=None, progress: float = 0.0) -> LossOutput:
        logits, _ = model(src_x)
        cls = F.cross_entropy(logits, src_y)
        return LossOutput(cls, make_logs(cls))