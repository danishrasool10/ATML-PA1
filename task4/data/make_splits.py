"""Stratified 90/10 split of CIFAR-10 train (seed 6304)."""
import argparse
from pathlib import Path

import numpy as np

SEED = 6304
VAL_FRACTION = 0.1


def stratified_split(labels, val_fraction=VAL_FRACTION, seed=SEED):
    """Return sorted train/val indices with class balance preserved."""
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        idx = idx[rng.permutation(len(idx))]
        n_val = int(round(len(idx) * val_fraction))
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    train_idx = np.sort(np.concatenate(train_idx))
    val_idx = np.sort(np.concatenate(val_idx))
    return train_idx, val_idx


def main():
    from torchvision.datasets import CIFAR10

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=str(Path(__file__).resolve().parent.parent / "datasets"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    ds = CIFAR10(root=args.data_root, train=True, download=True)
    train_idx, val_idx = stratified_split(ds.targets, seed=args.seed)
    out = Path(args.data_root) / f"cifar10_split_seed{args.seed}.npz"
    np.savez(out, train_idx=train_idx, val_idx=val_idx)
    print(f"train={len(train_idx)} val={len(val_idx)} saved to {out}")
    print("val class counts:", np.bincount(np.asarray(ds.targets)[val_idx]).tolist())


if __name__ == "__main__":
    main()