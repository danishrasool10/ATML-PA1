"""CDAN: discriminator sees g(x) = vec(f ⊗ p). No entropy conditioning, no detaching of f or p."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from task2.methods.dann import DANNMethod


class CDANMethod(DANNMethod):
    name = "cdan"

    def discriminator_in_dim(self, model) -> int:
        return model.feature_dim * model.num_classes  # 512 * 7 = 3584

    def discriminator_input(self, feats: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)  # gradients flow into both f and p
        return torch.bmm(feats.unsqueeze(2), probs.unsqueeze(1)).flatten(start_dim=1)