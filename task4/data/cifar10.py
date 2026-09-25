"""CIFAR-10 arrays, transforms, and datasets."""
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from data.make_splits import SEED, stratified_split

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class ArrayDataset(Dataset):
    """Dataset over uint8 HWC image arrays and integer labels."""

    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = np.asarray(labels).astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = Image.fromarray(self.images[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i])


def build_transform(train, randaugment=False, ra_num_ops=2, ra_magnitude=9):
    """Train: crop(32,pad 4) + flip [+ RandAugment] -> ToTensor -> Normalize. Eval: ToTensor -> Normalize."""
    ops = []
    if train:
        ops += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        if randaugment:
            ops.append(transforms.RandAugment(num_ops=ra_num_ops, magnitude=ra_magnitude))
    ops += [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    return transforms.Compose(ops)


def load_cifar10_arrays(data_root, train):
    ds = CIFAR10(root=str(data_root), train=train, download=True)
    return np.asarray(ds.data, dtype=np.uint8), np.asarray(ds.targets, dtype=np.int64)


def get_cifar10_datasets(data_root, randaugment=False, ra_num_ops=2, ra_magnitude=9, seed=SEED):
    """Return train, train_eval, val, and test datasets."""
    data_root = Path(data_root)
    tr_x, tr_y = load_cifar10_arrays(data_root, True)
    te_x, te_y = load_cifar10_arrays(data_root, False)
    train_idx, val_idx = stratified_split(tr_y, seed=seed)

    train_tf = build_transform(True, randaugment, ra_num_ops, ra_magnitude)
    eval_tf = build_transform(False)
    return {
        "train": ArrayDataset(tr_x[train_idx], tr_y[train_idx], train_tf),
        "train_eval": ArrayDataset(tr_x[train_idx], tr_y[train_idx], eval_tf),
        "val": ArrayDataset(tr_x[val_idx], tr_y[val_idx], eval_tf),
        "test": ArrayDataset(te_x, te_y, eval_tf),
    }