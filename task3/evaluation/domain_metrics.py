"""Domain metrics, aggregations, per-class accuracy, and confusion matrices."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels, preds = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        preds.append(logits.argmax(dim=1).cpu())
        labels.append(y.cpu())
    return torch.cat(labels).numpy(), torch.cat(preds).numpy()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, float]:
    labels = list(range(num_classes))
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "n": int(len(y_true)),
    }


def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> Dict[str, float]:
    y_true, y_pred = predict(model, loader, device)
    return compute_metrics(y_true, y_pred, num_classes)


def evaluate_domains(
    model: nn.Module, loaders: Mapping[str, DataLoader], device: torch.device, num_classes: int
) -> Dict[str, Dict[str, float]]:
    return {d: evaluate_loader(model, loader, device, num_classes) for d, loader in loaders.items()}


def aggregate_domain_metrics(per_domain: Mapping[str, Mapping[str, float]]) -> Dict[str, Any]:
    domains = list(per_domain.keys())
    accs = [per_domain[d]["acc"] for d in domains]
    f1s = [per_domain[d]["macro_f1"] for d in domains]
    return {
        "per_domain": {d: {"acc": float(per_domain[d]["acc"]), "macro_f1": float(per_domain[d]["macro_f1"])} for d in domains},
        "mean_acc": float(np.mean(accs)),
        "mean_macro_f1": float(np.mean(f1s)),
        "worst_acc": float(np.min(accs)),
        "worst_macro_f1": float(np.min(f1s)),
        "worst_domain_by_f1": domains[int(np.argmin(f1s))],
    }


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.full(num_classes, np.nan, dtype=np.float64)
    for c in range(num_classes):
        mask = y_true == c
        if mask.any():
            out[c] = float((y_pred[mask] == c).mean())
    return out


def confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))