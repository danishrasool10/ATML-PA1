"""DAN: source-classification loss + multi-kernel MMD."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from task2.methods.source_only import LossOutput, SourceOnlyMethod, make_logs

STUDY = {"param": "lambda_mmd", "tag": "lmmd", "values": [0.1, 1.0, 10.0]}

def multi_kernel_mmd(source: torch.Tensor, target: torch.Tensor, multipliers=(0.5, 1.0, 2.0)) -> torch.Tensor:
    """Biased MMD^2 with a sum of RBF kernels."""
    x = torch.cat([source, target], dim=0)
    n_s, n = source.size(0), x.size(0)
    d2 = (x.unsqueeze(1) - x.unsqueeze(0)).pow(2).sum(dim=-1)
    
    with torch.no_grad():
        iu = torch.triu_indices(n, n, offset=1, device=x.device)
        median = d2[iu[0], iu[1]].median().clamp_min(1e-8)
        
    k = sum(torch.exp(-d2 / (m * median)) for m in multipliers)
    k_ss, k_tt, k_st = k[:n_s, :n_s], k[n_s:, n_s:], k[:n_s, n_s:]
    return k_ss.mean() + k_tt.mean() - 2.0 * k_st.mean()

class DANMethod(SourceOnlyMethod):
    name = "dan"
    uses_target = True

    def __init__(self, cfg: dict, model):
        super().__init__(cfg, model)
        self.lambda_mmd = float(cfg.get("lambda_mmd", 1.0))
        self.multipliers = tuple(float(m) for m in cfg.get("bandwidth_multipliers", (0.5, 1.0, 2.0)))

    def compute_loss(self, model, src_x, src_y, tgt_x=None, progress: float = 0.0) -> LossOutput:
        n_s = src_x.size(0)
        logits, feats = model(torch.cat([src_x, tgt_x], dim=0))
        
        cls = F.cross_entropy(logits[:n_s], src_y)
        mmd = multi_kernel_mmd(feats[:n_s], feats[n_s:], self.multipliers)
        total = cls + self.lambda_mmd * mmd
        
        return LossOutput(total, make_logs(cls, mmd, total))