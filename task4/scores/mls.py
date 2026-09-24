import numpy as np


def mls_score(logits, features=None):
    """u_MLS = -max_k z_k."""
    return -np.asarray(logits, dtype=np.float64).max(axis=1)