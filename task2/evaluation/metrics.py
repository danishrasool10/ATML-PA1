"""Accuracy / macro-F1 utilities and inference helpers."""
from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(labels, preds, num_classes: int) -> Dict[str, float]:
    labels, preds = np.asarray(labels), np.asarray(preds)
    return {
        "acc": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, labels=list(range(num_classes)), average="macro", zero_division=0)),
        "n": int(len(labels)),
    }


@torch.no_grad()
def collect_outputs(model, loader, device, want_features: bool = False) -> Dict[str, np.ndarray]:
    model.eval()  # BN statistics are the frozen ImageNet ones, so eval() == training-time BN behaviour
    preds, labels, feats = [], [], []
    for images, targets, _ in loader:
        logits, f = model(images.to(device, non_blocking=True))
        preds.append(logits.argmax(dim=1).cpu())
        labels.append(targets)
        if want_features:
            feats.append(f.float().cpu())
    out = {"preds": torch.cat(preds).numpy(), "labels": torch.cat(labels).numpy()}
    if want_features:
        out["features"] = torch.cat(feats).numpy()
    return out


def evaluate_loader(model, loader, device, num_classes: int) -> Dict[str, float]:
    out = collect_outputs(model, loader, device)
    return classification_metrics(out["labels"], out["preds"], num_classes)


def evaluate_source_validation(model, val_loaders: Mapping[str, object], device, num_classes: int) -> Dict[str, object]:
    """Per-source-domain metrics plus mean_acc / mean_macro_f1 (mean over the three validation splits)."""
    result = {d: evaluate_loader(model, loader, device, num_classes) for d, loader in val_loaders.items()}
    means = {
        "mean_acc": float(np.mean([m["acc"] for m in result.values()])),
        "mean_macro_f1": float(np.mean([m["macro_f1"] for m in result.values()])),
    }
    result.update(means)
    return result