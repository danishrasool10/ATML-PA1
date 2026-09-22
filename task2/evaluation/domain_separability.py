"""Domain separability: held-out accuracy of a balanced logistic regression."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def domain_separability_score(
    source_features: np.ndarray,
    target_features: np.ndarray,
    seed: int = 6304,
    test_size: float = 0.3,
    C: float = 1.0,
    standardize: bool = False,
) -> Dict[str, float]:
    """Equal numbers of source-validation and target features -> 70/30 split -> LogReg(C=1, balanced)."""
    rng = np.random.RandomState(seed)
    n = min(len(source_features), len(target_features))
    src = source_features[np.sort(rng.choice(len(source_features), n, replace=False))]
    tgt = target_features[np.sort(rng.choice(len(target_features), n, replace=False))]

    X = np.concatenate([src, tgt]).astype(np.float64)
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    if standardize:
        scaler = StandardScaler().fit(X_tr)
        X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
    clf.fit(X_tr, y_tr)
    return {
        "accuracy": float(clf.score(X_te, y_te)),
        "train_accuracy": float(clf.score(X_tr, y_tr)),
        "n_per_domain": int(n),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "chance": 0.5,
    }