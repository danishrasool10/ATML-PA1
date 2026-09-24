"""Empirical Risk Minimization baseline across source domains."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping

import torch
import torch.nn.functional as F
from torch import nn


def balanced_ce(logits: torch.Tensor, y: torch.Tensor, sizes: List[int]) -> torch.Tensor:
    per_domain = [F.cross_entropy(l, t) for l, t in zip(torch.split(logits, sizes), torch.split(y, sizes))]
    return torch.stack(per_domain).mean()


class ERMMethod:
    name = "erm"

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, cfg: Mapping[str, Any]) -> None:
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg

    def train_step(self, x: torch.Tensor, y: torch.Tensor, sizes: List[int]) -> Dict[str, torch.Tensor]:
        logits = self.model(x)
        loss = balanced_ce(logits, y, sizes)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.detach(), "cls_loss": loss.detach()}


def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    if isinstance(obj, dict):
        for key in ("model_state_dict", "model", "state_dict", "net", "weights"):
            if key in obj and isinstance(obj[key], (dict, nn.Module)):
                return _extract_state_dict(obj[key])
        if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
            return dict(obj)
    raise ValueError("Could not find a state dict in the checkpoint.")


def _remap(state: Dict[str, torch.Tensor], backbone_prefix: str) -> Dict[str, torch.Tensor]:
    head_prefixes = sorted(["head.fc.", "classifier.fc.", "fc.", "classifier.", "head."], key=len, reverse=True)
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        hp = next((p for p in head_prefixes if k.startswith(p)), None)
        if hp is not None:
            out["head.fc." + k[len(hp):]] = v
        elif k.startswith(backbone_prefix):
            out["backbone.net." + k[len(backbone_prefix):]] = v
    return out


def load_task2_checkpoint(model: nn.Module, path: str, device: torch.device) -> str:
    """Load pretrained Task 2 source-only checkpoint into DGModel."""
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(path, map_location="cpu")
    state = _extract_state_dict(raw)
    state = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state.items()}

    candidates = [("identity", state)]
    if any(k.startswith("model.") for k in state):
        candidates.append(("strip 'model.'", {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}))
    for bp in ("", "backbone.", "encoder.", "feature_extractor.", "model."):
        candidates.append((f"remap backbone_prefix='{bp}'", _remap(state, bp)))

    errors = []
    for name, cand in candidates:
        try:
            model.load_state_dict(cand, strict=True)
            model.to(device)
            return name
        except RuntimeError as exc:
            errors.append(f"[{name}] {str(exc).splitlines()[0]}")
    sample = list(state.keys())[:5]
    raise RuntimeError(
        "Could not load the Task 2 checkpoint into DGModel.\nSample keys: "
        f"{sample}\nAttempts:\n" + "\n".join(errors)
    )