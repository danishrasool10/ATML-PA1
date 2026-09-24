"""CIFAR-100 unknowns. EVALUATION ONLY: only the official test partition is ever read,
and access requires an explicit allow_eval_access=True (used only by final extraction / evaluate_osr.py)."""
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR100

NEAR_CLASSES = ["bus", "pickup_truck", "motorcycle", "tractor", "wolf", "fox", "leopard", "camel"]
FAR_CLASSES = ["bottle", "bowl", "chair", "clock", "keyboard", "mushroom", "sunflower", "wardrobe"]
GROUP_NEAR, GROUP_FAR = 0, 1


def load_unknowns(data_root, allow_eval_access=False):
    """Returns dict: images (N,32,32,3 uint8), fine_labels (N,), group (N,; 0=near, 1=far), fine_names (100,)."""
    if not allow_eval_access:
        raise RuntimeError("CIFAR-100 unknowns are evaluation-only; pass allow_eval_access=True in final evaluation.")
    ds = CIFAR100(root=str(Path(data_root)), train=False, download=True)  # test partition only
    name_to_idx = {n: i for i, n in enumerate(ds.classes)}
    data = np.asarray(ds.data)
    targets = np.asarray(ds.targets)
    images, fine, group = [], [], []
    for g, names in ((GROUP_NEAR, NEAR_CLASSES), (GROUP_FAR, FAR_CLASSES)):
        for n in names:
            idx = np.where(targets == name_to_idx[n])[0]
            images.append(data[idx])
            fine.append(np.full(len(idx), name_to_idx[n], dtype=np.int64))
            group.append(np.full(len(idx), g, dtype=np.int64))
    out = {
        "images": np.concatenate(images).astype(np.uint8),
        "fine_labels": np.concatenate(fine),
        "group": np.concatenate(group),
        "fine_names": np.asarray(ds.classes),
    }
    assert int((out["group"] == GROUP_NEAR).sum()) == 800 and int((out["group"] == GROUP_FAR).sum()) == 800
    return out
