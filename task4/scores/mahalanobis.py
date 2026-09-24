import numpy as np


class MahalanobisScorer:
    """Mahalanobis score with a shared diagonal covariance (pooled within-class variance) and eps on the diagonal."""

    def __init__(self, eps=1e-6):
        self.eps = eps
        self.means = None
        self.var = None

    def fit(self, features, labels):
        f = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels)
        classes = np.unique(labels)
        self.means = np.stack([f[labels == c].mean(axis=0) for c in classes])
        centered = f - self.means[np.searchsorted(classes, labels)]
        self.var = (centered ** 2).mean(axis=0) + self.eps
        return self

    def score(self, features):
        f = np.asarray(features, dtype=np.float64)
        inv = 1.0 / self.var
        d = np.stack([(((f - mu) ** 2) * inv).sum(axis=1) for mu in self.means], axis=1)
        return d.min(axis=1)

    def __call__(self, logits, features):
        return self.score(features)


def fit_mahalanobis(train_features, train_labels):
    """Fit on unaugmented CIFAR-10 training-portion features; returns a callable (logits, features) -> u."""
    return MahalanobisScorer().fit(train_features, train_labels)