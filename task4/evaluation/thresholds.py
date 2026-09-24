import numpy as np


def calibrate_threshold(val_known_scores, accept_rate=0.95):
    """tau = 95th percentile of validation unknownness."""
    return float(np.percentile(np.asarray(val_known_scores, dtype=np.float64), accept_rate * 100.0))


def accept(scores, tau):
    """Accept x as known when u(x) <= tau."""
    return np.asarray(scores) <= tau