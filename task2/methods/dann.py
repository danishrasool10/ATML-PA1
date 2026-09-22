"""DANN: adversarial marginal alignment with a gradient-reversal domain discriminator."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from task2.methods.source_only import LossOutput, SourceOnlyMethod, make_logs
from task2.models.domain_discriminator import DomainDiscriminator, grl_schedule

STUDY = {"param": "grl_max", "tag": "grlmax", "values": [0.25, 0.5, 1.0]}


class DANNMethod(SourceOnlyMethod):
    name = "dann"
    uses_target = True

    def __init__(self, cfg: dict, model):
        super().__init__(cfg, model)
        self.grl_max = float(cfg.get("grl_max", 1.0))
        self.grl_gamma = float(cfg.get("grl_gamma", 10.0))
        self.domain_weight = float(cfg.get("domain_loss_weight", 1.0))
        self.discriminator = DomainDiscriminator(
            self.discriminator_in_dim(model),
            hidden=int(cfg.get("disc_hidden", 256)),
            dropout=float(cfg.get("disc_dropout", 0.5)),
        )

    def discriminator_in_dim(self, model) -> int:
        return model.feature_dim

    def discriminator_input(self, feats: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        return feats

    def compute_loss(self, model, src_x, src_y, tgt_x=None, progress: float = 0.0) -> LossOutput:
        n_s, n_t = src_x.size(0), tgt_x.size(0)
        logits, feats = model(torch.cat([src_x, tgt_x], dim=0))
        cls = F.cross_entropy(logits[:n_s], src_y)

        alpha = grl_schedule(progress, self.grl_gamma, self.grl_max)
        d_logits = self.discriminator(self.discriminator_input(feats, logits), alpha)
        d_labels = torch.cat([
            torch.zeros(n_s, dtype=torch.long, device=feats.device),
            torch.ones(n_t, dtype=torch.long, device=feats.device),
        ])
        dom = F.cross_entropy(d_logits, d_labels)
        total = cls + self.domain_weight * dom
        dom_acc = float((d_logits.argmax(dim=1) == d_labels).float().mean())
        return LossOutput(total, make_logs(cls, dom, total, dom_acc, alpha))