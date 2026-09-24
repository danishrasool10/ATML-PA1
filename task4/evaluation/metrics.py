import numpy as np
from sklearn.metrics import roc_auc_score

from evaluation.thresholds import accept, calibrate_threshold


def auroc(known_scores, unknown_scores):
    """AUROC with unknown as the positive class."""
    y = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    s = np.concatenate([known_scores, unknown_scores])
    return float(roc_auc_score(y, s))


def closed_set_accuracy(logits, labels, num_known=10):
    """Accuracy using only the first `num_known` logits."""
    return float((np.asarray(logits)[:, :num_known].argmax(axis=1) == np.asarray(labels)).mean())


def osr_metrics(val_known, test_known, near, far, accept_rate=0.95):
    """Return AUROC and validation-calibrated rejection metrics."""
    tau = calibrate_threshold(val_known, accept_rate)
    allu = np.concatenate([near, far])
    out = {"tau": tau,
           "known_accept": float(accept(test_known, tau).mean()),
           "auroc_near": auroc(test_known, near),
           "auroc_far": auroc(test_known, far),
           "auroc_all": auroc(test_known, allu)}
    for name, s in (("near", near), ("far", far), ("all", allu)):
        fpr = float(accept(s, tau).mean())
        out[f"fpr_{name}"] = fpr
        out[f"reject_{name}"] = 1.0 - fpr
    return out