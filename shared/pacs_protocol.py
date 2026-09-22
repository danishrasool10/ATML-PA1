"""Shared PACS protocol (Tasks 2 and 3): seeding, config helpers, stratified splits, samplers, loaders."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader

from shared.pacs import (
    CLASS_NAMES,
    SOURCE_DOMAINS,
    TARGET_DOMAIN,
    PACSDataset,
    build_transforms,
    list_domain_samples,
    resolve_pacs_root,
)

SEED = 6304
VAL_FRACTION = 0.2
DEFAULT_SPLIT_FILE = REPO_ROOT / "shared" / "splits" / "pacs_sketch_seed6304.json"

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def resolve_repo_path(path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path

def deep_update(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (new or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out

def load_yaml(path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def load_config(config_path, overrides: dict | None = None) -> dict:
    """base.yaml (same folder) <- method yaml <- overrides."""
    config_path = Path(config_path)
    cfg: dict = {}
    if config_path.name != "base.yaml":
        base_path = config_path.parent / "base.yaml"
        if base_path.exists():
            cfg = load_yaml(base_path)
    cfg = deep_update(cfg, load_yaml(config_path))
    return deep_update(cfg, overrides or {})

def parse_overrides(pairs: Sequence[str]) -> Dict[str, Any]:
    """['method.lambda_mmd=10', 'train.max_epochs=5'] -> nested dict."""
    out: Dict[str, Any] = {}
    for item in pairs or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"Override must look like a.b.c=value, got '{item}'")
        *parents, leaf = key.split(".")
        node = out
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return out

def build_splits(root, seed: int = SEED, val_fraction: float = VAL_FRACTION) -> dict:
    """Stratified 80/20 train/val split inside every source domain; the full target domain is listed as well."""
    root = Path(root)
    splits: dict = {
        "seed": seed,
        "val_fraction": val_fraction,
        "target_domain": TARGET_DOMAIN,
        "source_domains": list(SOURCE_DOMAINS),
        "class_names": list(CLASS_NAMES),
        "sources": {},
        "target": None,
    }
    for domain in SOURCE_DOMAINS:
        samples = list_domain_samples(root, domain)
        labels = np.array([y for _, y in samples])
        train_idx, val_idx = train_test_split(
            np.arange(len(samples)), test_size=val_fraction, random_state=seed, stratify=labels
        )
        splits["sources"][domain] = {
            "train": [[samples[i][0], int(samples[i][1])] for i in sorted(train_idx.tolist())],
            "val": [[samples[i][0], int(samples[i][1])] for i in sorted(val_idx.tolist())],
        }
    splits["target"] = [[p, int(y)] for p, y in list_domain_samples(root, TARGET_DOMAIN)]
    return splits

def _verify_files(root: Path, splits: dict) -> None:
    entries = [e for d in SOURCE_DOMAINS for part in ("train", "val") for e in splits["sources"][d][part]]
    entries += splits["target"]
    missing = [rel for rel, _ in entries if not (root / rel).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} files listed in the split file are missing under {root} (e.g. {missing[:3]}). "
            "Run `python -m shared.pacs_protocol --root <PACS> --force` to rebuild the split file."
        )

def load_or_create_splits(root, split_file=DEFAULT_SPLIT_FILE, seed: int = SEED) -> dict:
    root, split_file = Path(root), Path(split_file)
    if split_file.exists():
        with open(split_file, "r") as f:
            splits = json.load(f)
        if splits.get("sources"):
            if int(splits["seed"]) != int(seed):
                raise ValueError(f"Split file seed {splits['seed']} != requested seed {seed}")
            _verify_files(root, splits)
            return splits
    splits = build_splits(root, seed)
    split_file.parent.mkdir(parents=True, exist_ok=True)
    with open(split_file, "w") as f:
        json.dump(splits, f)
    print(f"[splits] wrote {split_file}")
    return splits

def build_source_datasets(root, splits: dict, image_size: int = 256, crop_size: int = 224):
    train_tf = build_transforms(True, image_size, crop_size)
    eval_tf = build_transforms(False, image_size, crop_size)
    train = {d: PACSDataset(root, splits["sources"][d]["train"], d, train_tf) for d in SOURCE_DOMAINS}
    val = {d: PACSDataset(root, splits["sources"][d]["val"], d, eval_tf) for d in SOURCE_DOMAINS}
    return train, val

def build_target_adaptation_dataset(root, splits: dict, image_size: int = 256, crop_size: int = 224):
    return PACSDataset(root, splits["target"], TARGET_DOMAIN, build_transforms(True, image_size, crop_size), use_labels=False)

def build_target_eval_dataset(root, splits: dict, image_size: int = 256, crop_size: int = 224):
    return PACSDataset(root, splits["target"], TARGET_DOMAIN, build_transforms(False, image_size, crop_size), use_labels=True)

class DomainBalancedBatchSampler:
    """Index batches with exactly `per_domain` examples from every domain of a ConcatDataset."""

    def __init__(self, domain_sizes: Sequence[int], per_domain: int, steps_per_epoch: int, seed: int):
        self.sizes = [int(s) for s in domain_sizes]
        if any(s < per_domain for s in self.sizes):
            raise ValueError("Every source domain needs at least `per_domain` training images.")
        self.offsets = np.cumsum([0] + self.sizes[:-1]).tolist()
        self.per_domain = int(per_domain)
        self.steps_per_epoch = int(steps_per_epoch)
        self._rngs = [np.random.RandomState(seed + k) for k in range(len(self.sizes))]
        self._perms = [rng.permutation(n) for rng, n in zip(self._rngs, self.sizes)]
        self._ptr = [0] * len(self.sizes)

    def _draw(self, k: int) -> List[int]:
        if self._ptr[k] + self.per_domain > self.sizes[k]:
            self._perms[k] = self._rngs[k].permutation(self.sizes[k])
            self._ptr[k] = 0
        chunk = self._perms[k][self._ptr[k]: self._ptr[k] + self.per_domain]
        self._ptr[k] += self.per_domain
        return (chunk + self.offsets[k]).tolist()

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.steps_per_epoch):
            batch: List[int] = []
            for k in range(len(self.sizes)):
                batch.extend(self._draw(k))
            yield batch

    def __len__(self) -> int:
        return self.steps_per_epoch

def default_steps_per_epoch(train_datasets: dict, per_domain_batch: int) -> int:
    total = sum(len(ds) for ds in train_datasets.values())
    return math.ceil(total / (per_domain_batch * len(train_datasets)))

def build_source_loader(train_datasets: dict, per_domain_batch: int, steps_per_epoch: int, seed: int, num_workers: int) -> DataLoader:
    datasets = [train_datasets[d] for d in SOURCE_DOMAINS]
    sampler = DomainBalancedBatchSampler([len(d) for d in datasets], per_domain_batch, steps_per_epoch, seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        ConcatDataset(datasets), batch_sampler=sampler, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), generator=gen, worker_init_fn=_seed_worker,
    )

def build_target_loader(dataset, batch_size: int, seed: int, num_workers: int) -> DataLoader:
    gen = torch.Generator()
    gen.manual_seed(seed + 1000)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), generator=gen, worker_init_fn=_seed_worker,
    )

def build_eval_loader(dataset, batch_size: int = 128, num_workers: int = 4) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), worker_init_fn=_seed_worker,
    )

def cycle_loader(loader: DataLoader) -> Iterator:
    while True:
        for batch in loader:
            yield batch

def main() -> None:
    parser = argparse.ArgumentParser(description="Create / inspect the shared PACS split file.")
    parser.add_argument("--root", required=True, help="PACS folder (contains photo/, art_painting/, cartoon/, sketch/)")
    parser.add_argument("--split_file", default=str(DEFAULT_SPLIT_FILE))
    parser.add_argument("--force", action="store_true", help="rebuild even if the split file is populated")
    args = parser.parse_args()

    root = resolve_pacs_root(args.root)
    if args.force:
        splits = build_splits(root)
        Path(args.split_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.split_file, "w") as f:
            json.dump(splits, f)
    else:
        splits = load_or_create_splits(root, args.split_file)
    for d in SOURCE_DOMAINS:
        print(f"{d:>13}: train {len(splits['sources'][d]['train']):5d} | val {len(splits['sources'][d]['val']):4d}")
    print(f"{TARGET_DOMAIN:>13}: {len(splits['target'])} images (unlabeled during Task 2 adaptation)")

if __name__ == "__main__":
    main()