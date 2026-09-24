import numpy as np


def msp_score(logits, features=None):
    """u_MSP = 1 - max_k softmax_k, computed as r / (1 + r) to avoid exact 0."""
    z = np.asarray(logits, dtype=np.float64)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    e[np.arange(len(e)), z.argmax(axis=1)] = 0.0
    r = e.sum(axis=1)
    return r / (1.0 + r)