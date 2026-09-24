"""Source domain separability via multinomial logistic regression on frozen features."""
from __future__ import annotations

import warnings
from typing import Any, Dict, Mapping

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


@torch.no_grad()
def extract_features(
    model: nn.Module, dataset: Dataset, device: torch.device, batch_size: int = 64, num_workers: int = 0
) -> np.ndarray:
    model.eval()
    backbone = model.backbone if hasattr(model, "backbone") else model
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    feats = []
    for x, _ in loader:
        feats.append(backbone(x.to(device, non_blocking=True)).cpu())
    return torch.cat(feats).numpy()


def _fit_logreg(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            clf = LogisticRegression(C=1.0, max_iter=1000, multi_class="multinomial")
        except TypeError:
            clf = LogisticRegression(C=1.0, max_iter=1000)
        clf.fit(X_train, y_train)
    return clf


def source_domain_separability(
    model: nn.Module,
    balanced_val_sets: Mapping[str, Dataset],
    device: torch.device,
    seed: int = 6304,
    train_frac: float = 0.7,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Dict[str, Any]:
    X_parts, y_parts = [], []
    for domain_id, (_, ds) in enumerate(balanced_val_sets.items()):
        feats = extract_features(model, ds, device, batch_size, num_workers)
        X_parts.append(feats)
        y_parts.append(np.full(len(feats), domain_id, dtype=np.int64))
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=train_frac, random_state=seed, stratify=y)
    clf = _fit_logreg(X_tr, y_tr)
    acc = float(clf.score(X_te, y_te))
    return {
        "accuracy": acc,
        "chance": 1.0 / len(balanced_val_sets),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "domains": list(balanced_val_sets.keys()),
    }