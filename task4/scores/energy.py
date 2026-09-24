import numpy as np


def energy_score(logits, features=None):
    """u_Energy = -log sum_k exp(z_k) (temperature 1)."""
    z = np.asarray(logits, dtype=np.float64)
    m = z.max(axis=1, keepdims=True)
    return -(m.squeeze(1) + np.log(np.exp(z - m).sum(axis=1)))