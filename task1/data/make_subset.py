import json
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import STL10

SEED = 6304
VAL_FRACTION = 0.2
TEST_SUBSET_SIZE = 500

def _stratified_indices(labels, val_fraction, seed):
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        n_val = int(round(len(idx_c) * val_fraction))
        val_idx.extend(idx_c[:n_val].tolist())
        train_idx.extend(idx_c[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return sorted(train_idx), sorted(val_idx)

def _balanced_test_subset(labels, n_total, seed):
    rng = np.random.RandomState(seed)
    classes = np.unique(labels)
    n_per_class = n_total // len(classes)
    chosen, imbalance = [], {}
    for c in classes:
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        take = min(n_per_class, len(idx_c))
        if take < n_per_class:
            imbalance[int(c)] = {"requested": n_per_class, "available": int(len(idx_c))}
        chosen.extend(idx_c[:take].tolist())
    return sorted(chosen), imbalance

def build_splits(data_root, out_dir, seed=SEED):
    os.makedirs(out_dir, exist_ok=True)
    train_ds = STL10(root=data_root, split="train", download=True)
    test_ds = STL10(root=data_root, split="test", download=True)

    train_labels = np.asarray(train_ds.labels)
    test_labels = np.asarray(test_ds.labels)

    train_idx, val_idx = _stratified_indices(train_labels, VAL_FRACTION, seed)
    test_subset_idx, imbalance = _balanced_test_subset(test_labels, TEST_SUBSET_SIZE, seed)

    splits = {
        "seed": seed,
        "classes": list(train_ds.classes),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_subset_idx": test_subset_idx,
        "test_subset_size_requested": TEST_SUBSET_SIZE,
        "test_subset_size_actual": len(test_subset_idx),
        "test_subset_class_imbalance": imbalance,
    }
    
    with open(os.path.join(out_dir, "splits.json"), "w") as f:
        json.dump(splits, f, indent=2)

    print(f"train={len(train_idx)} val={len(val_idx)} test_subset={len(test_subset_idx)}")
    if imbalance:
        print(f"WARNING: class imbalance in test subset: {imbalance}")
        
    return splits

def load_splits(out_dir):
    with open(os.path.join(out_dir, "splits.json")) as f:
        return json.load(f)

class STL10Subset(Dataset):
    def __init__(self, stl10_dataset, indices, transform):
        self.ds = stl10_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        real_idx = self.indices[i]
        img, label = self.ds[real_idx]
        img = self.transform(img)
        return img, label, real_idx

if __name__ == "__main__":
    build_splits(data_root="./data/raw", out_dir="./data/splits")