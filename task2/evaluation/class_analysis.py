"""Per-class target analysis."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def confusion_matrix_np(labels, preds, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(labels), np.asarray(preds)), 1)
    return cm  # rows = true class, cols = predicted class


def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    support = cm.sum(axis=1)
    return np.divide(np.diag(cm), support, out=np.zeros(len(support), dtype=np.float64), where=support > 0)


def dominant_confusion(cm: np.ndarray, class_idx: int, class_names: Sequence[str]) -> Optional[Dict]:
    row = cm[class_idx].astype(np.float64).copy()
    total = row.sum()
    row[class_idx] = -1.0
    j = int(np.argmax(row))
    if total == 0 or row[j] <= 0:
        return None
    return {"true": class_names[class_idx], "pred": class_names[j], "count": int(row[j]), "rate": float(row[j] / total)}


def top_confusions(cm: np.ndarray, class_names: Sequence[str], k: int = 5) -> List[Dict]:
    off = cm.copy()
    np.fill_diagonal(off, 0)
    support = cm.sum(axis=1)
    items = []
    for flat in np.argsort(off, axis=None)[::-1][:k]:
        i, j = np.unravel_index(flat, off.shape)
        if off[i, j] == 0:
            break
        items.append({"true": class_names[i], "pred": class_names[j], "count": int(off[i, j]),
                      "rate": float(off[i, j] / support[i])})
    return items


def compare_to_baseline(cm_base: np.ndarray, cm_method: np.ndarray, class_names: Sequence[str]) -> Dict:
    acc_b, acc_m = per_class_accuracy(cm_base), per_class_accuracy(cm_method)
    delta = acc_m - acc_b
    per_class = [
        {"class": n, "baseline_acc": float(acc_b[i]), "method_acc": float(acc_m[i]),
         "delta_pp": float(100.0 * delta[i]), "support": int(cm_base[i].sum())}
        for i, n in enumerate(class_names)
    ]

    def describe(i: int) -> Dict:
        return {
            "class": class_names[i],
            "delta_pp": float(100.0 * delta[i]),
            "baseline_dominant_confusion": dominant_confusion(cm_base, i, class_names),
            "method_dominant_confusion": dominant_confusion(cm_method, i, class_names),
        }

    i_up, i_down = int(np.argmax(delta)), int(np.argmin(delta))
    return {
        "per_class": per_class,
        "largest_improvement": describe(i_up),
        "largest_degradation": {**describe(i_down), "is_negative_transfer": bool(delta[i_down] < 0)},
        "num_classes_degraded": int((delta < 0).sum()),
    }


def flip_statistics(labels, base_preds, method_preds, class_names: Sequence[str]) -> Dict:
    labels, base_preds, method_preds = map(np.asarray, (labels, base_preds, method_preds))
    helped = (base_preds != labels) & (method_preds == labels)
    harmed = (base_preds == labels) & (method_preds != labels)
    return {
        "helped": int(helped.sum()),
        "harmed": int(harmed.sum()),
        "per_class": [
            {"class": n, "helped": int((helped & (labels == i)).sum()), "harmed": int((harmed & (labels == i)).sum())}
            for i, n in enumerate(class_names)
        ],
    }


def flip_examples(labels, base_preds, method_preds, class_idx: int, kind: str = "harmed", k: int = 5) -> List[int]:
    """Indices of target samples of one class that the method flipped relative to Source-only."""
    labels, base_preds, method_preds = map(np.asarray, (labels, base_preds, method_preds))
    base_ok, meth_ok = base_preds == labels, method_preds == labels
    flipped = (base_ok & ~meth_ok) if kind == "harmed" else (~base_ok & meth_ok)
    return np.flatnonzero((labels == class_idx) & flipped)[:k].tolist()