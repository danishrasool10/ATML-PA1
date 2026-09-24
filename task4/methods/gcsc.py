"""GCSC: Vanilla plus RandAugment(num_ops=2, magnitude=9)."""
from methods.vanilla import run_training


def run(cfg):
    cfg = dict(cfg)
    aug = dict(cfg.get("augmentation", {}))
    aug["randaugment"] = True
    aug.setdefault("ra_num_ops", 2)
    aug.setdefault("ra_magnitude", 9)
    cfg["augmentation"] = aug
    return run_training(cfg, randaugment=True)