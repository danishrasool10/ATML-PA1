"""Extract and cache features/logits from best checkpoints."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.cifar10 import ArrayDataset, build_transform, get_cifar10_datasets
from data.cifar100_unknowns import load_unknowns
from models.resnet_cifar import ResNet18Cifar

ROOT = Path(__file__).resolve().parent
METHODS = ["vanilla", "gcsc", "proser"]
KNOWN_SPLITS = ["train", "val", "test"]
SEED = 6304


def ckpt_path(ckpt_dir, method):
    return Path(ckpt_dir) / method / "best.pt"


def cache_path(cache_dir, method, split):
    return Path(cache_dir) / method / f"{split}.npz"


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu")
    model = ResNet18Cifar(num_classes=int(ckpt["num_classes"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def run_model(model, dataset, device, batch_size=500, num_workers=4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=device.type == "cuda")
    feats, logits, labels = [], [], []
    for x, y in loader:
        lg, ft = model(x.to(device, non_blocking=True), return_features=True)
        feats.append(ft.float().cpu().numpy())
        logits.append(lg.float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(logits), np.concatenate(labels)


def ensure_known_cache(method, ckpt_dir, cache_dir, data_root, device, batch_size=500,
                       num_workers=4, overwrite=False):
    paths = {s: cache_path(cache_dir, method, s) for s in KNOWN_SPLITS}
    if not overwrite and all(p.exists() for p in paths.values()):
        return
    model = load_model(ckpt_path(ckpt_dir, method), device)
    ds = get_cifar10_datasets(data_root, randaugment=False, seed=SEED)
    for split, key in (("train", "train_eval"), ("val", "val"), ("test", "test")):
        f, z, y = run_model(model, ds[key], device, batch_size, num_workers)
        paths[split].parent.mkdir(parents=True, exist_ok=True)
        np.savez(paths[split], features=f, logits=z, labels=y)
        print(f"[extract] {method}/{split}: features {f.shape} logits {z.shape}")


def ensure_unknown_cache(method, ckpt_dir, cache_dir, data_root, device, batch_size=500,
                         num_workers=4, overwrite=False):
    missing = [m for m in METHODS if not ckpt_path(ckpt_dir, m).exists()]
    if missing:
        raise RuntimeError(f"CIFAR-100 unknowns may only be touched after all checkpoints are final; "
                           f"missing checkpoints for: {missing}")
    path = cache_path(cache_dir, method, "unknown")
    if path.exists() and not overwrite:
        return
    u = load_unknowns(data_root, allow_eval_access=True)
    ds = ArrayDataset(u["images"], u["group"], build_transform(False))
    model = load_model(ckpt_path(ckpt_dir, method), device)
    f, z, g = run_model(model, ds, device, batch_size, num_workers)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, features=f, logits=z, group=g, fine_labels=u["fine_labels"], fine_names=u["fine_names"])
    print(f"[extract] {method}/unknown: features {f.shape} logits {z.shape}")


def load_cache(cache_dir, method, splits):
    out = {}
    for s in splits:
        with np.load(cache_path(cache_dir, method, s)) as d:
            out[s] = {k: d[k] for k in d.files}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--unknowns", action="store_true", help="also extract CIFAR-100 unknowns (final evaluation only)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--data_root", type=str, default=str(ROOT / "datasets"))
    parser.add_argument("--ckpt_dir", type=str, default=str(ROOT / "checkpoints"))
    parser.add_argument("--cache_dir", type=str, default=str(ROOT / "cache"))
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for m in args.methods:
        ensure_known_cache(m, args.ckpt_dir, args.cache_dir, args.data_root, device,
                           args.batch_size, args.num_workers, args.overwrite)
        if args.unknowns:
            ensure_unknown_cache(m, args.ckpt_dir, args.cache_dir, args.data_root, device,
                                 args.batch_size, args.num_workers, args.overwrite)


if __name__ == "__main__":
    main()