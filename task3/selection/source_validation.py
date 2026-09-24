"""Source data loaders, validation, and checkpoint selection."""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

import shared.pacs as shared_pacs
import shared.pacs_protocol as shared_protocol

from task3.evaluation.domain_metrics import aggregate_domain_metrics, evaluate_domains

TARGET_DOMAIN = "sketch"
DEFAULT_CLASS_NAMES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_DOMAIN_KEYS = {
    "photo": "photo",
    "artpainting": "art_painting",
    "art": "art_painting",
    "cartoon": "cartoon",
    "sketch": "sketch",
    "target": "sketch",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(p: str) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else repo_root() / path


def shared_attr(names: Sequence[str], default: Any = None) -> Any:
    for module in (shared_protocol, shared_pacs):
        for n in names:
            if hasattr(module, n):
                return getattr(module, n)
    return default


def get_class_names() -> List[str]:
    names = shared_attr(("CLASS_NAMES", "CLASSES", "PACS_CLASSES"))
    if names is not None:
        try:
            names = [str(n).lower() for n in names]
        except TypeError:
            names = None
    return names if names and len(names) == 7 else list(DEFAULT_CLASS_NAMES)


def resolve_data_root(cfg_root: Optional[str] = None) -> Optional[Path]:
    for cand in (cfg_root, os.environ.get("PACS_ROOT")):
        if cand:
            return Path(str(cand)).expanduser()
    return None


def canon_domain(name: Any) -> str:
    key = re.sub(r"[^a-z]", "", str(name).lower())
    if "photo" in key: return "photo"
    if "art" in key: return "art_painting"
    if "cartoon" in key: return "cartoon"
    if "sketch" in key or "target" in key: return "sketch"
    return _DOMAIN_KEYS.get(key, key)


def canon_part(name: Any) -> str:
    key = re.sub(r"[^a-z]", "", str(name).lower())
    if "train" in key: return "train"
    if "val" in key or "test" in key: return "val"
    return key


def find_domain_dir(root: Optional[Path], domain: str) -> Optional[Path]:
    """Locate domain folder under data root."""
    if root is None or not root.is_dir():
        return None
    bases = [root] + sorted(c for c in root.iterdir() if c.is_dir())
    for base in bases:
        for c in sorted(base.iterdir()):
            if c.is_dir() and canon_domain(c.name) == domain:
                return c
    return None


def normalize_entry(entry: Any, class_names: Sequence[str]) -> Tuple[str, int]:
    path: Any = None
    label: Any = None
    if isinstance(entry, dict):
        for k in ("path", "file", "filepath", "filename", "image", "img", "image_path"):
            if k in entry:
                path = entry[k]
                break
        for k in ("label", "class", "y", "target", "class_id"):
            if k in entry:
                label = entry[k]
                break
    elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
        path, label = entry[0], entry[1]
        if not isinstance(path, str) and isinstance(label, str):
            path, label = label, path
    elif isinstance(entry, str):
        path = entry
    if not isinstance(path, str):
        raise ValueError(f"Unsupported split entry: {entry!r}")
    if label is None:
        label = Path(path).parent.name
    if isinstance(label, str):
        low = label.strip().lower()
        label = class_names.index(low) if low in class_names else int(low)
    return path, int(label)


def resolve_sample_path(p: str, root: Optional[Path], domain_dir: Optional[Path]) -> str:
    pp = Path(p)
    cands = [pp]
    if root is not None:
        cands += [root / pp, root.parent / pp]
    if domain_dir is not None:
        cands += [domain_dir / pp, domain_dir.parent / pp]
    for c in cands:
        if c.is_file():
            return str(c)
    raise FileNotFoundError(f"Image '{p}' not found (data_root={root}). Set --data_root or $PACS_ROOT.")


def lookup_split(blob: Any, domain: str, part: str) -> Optional[list]:
    if isinstance(blob, dict):
        # 1. {domain: {part: [...]}}
        for k, v in blob.items():
            if canon_domain(k) == domain and isinstance(v, dict):
                for pk, pv in v.items():
                    if canon_part(pk) == part and isinstance(pv, list):
                        return pv
        # 2. {part: {domain: [...]}}
        for k, v in blob.items():
            if canon_part(k) == part and isinstance(v, dict):
                for dk, dv in v.items():
                    if canon_domain(dk) == domain and isinstance(dv, list):
                        return dv
        # 3. Flattened keys like "photo_train": [...]
        for k, v in blob.items():
            if isinstance(v, list) and canon_domain(k) == domain and canon_part(k) == part:
                return v
        # 4. {domain: [{"split": "train", "path": ...}, ...]}
        for k, v in blob.items():
            if canon_domain(k) == domain and isinstance(v, list) and v and isinstance(v[0], dict):
                rows = [e for e in v if canon_part(e.get("split", e.get("part", ""))) == part]
                if rows: return rows
        # 5. Deep search fallback
        for v in blob.values():
            if isinstance(v, (dict, list)):
                r = lookup_split(v, domain, part)
                if r is not None:
                    return r

    elif isinstance(blob, list) and blob and isinstance(blob[0], dict):
        # 6. Flat list of dicts: [{"domain": "photo", "split": "train", ...}]
        rows = [
            e for e in blob
            if canon_domain(e.get("domain", e.get("d", ""))) == domain and canon_part(e.get("split", e.get("part", ""))) == part
        ]
        if rows: return rows

    return None


def lookup_domain_all(blob: Any, domain: str) -> Optional[list]:
    if isinstance(blob, dict):
        for k, v in blob.items():
            if canon_domain(k) == domain:
                if isinstance(v, list):
                    return v
                if isinstance(v, dict):
                    out: list = []
                    for vv in v.values():
                        if isinstance(vv, list):
                            out += vv
                    if out:
                        return out
        for v in blob.values():
            if isinstance(v, dict):
                r = lookup_domain_all(v, domain)
                if r is not None:
                    return r
    return None


class PACSDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[str, int]], transform: Any) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        return self.transform(img), label


def load_source_splits(cfg: Mapping[str, Any]) -> Dict[str, Dict[str, List[Tuple[str, int]]]]:
    domains = [canon_domain(d) for d in cfg["source_domains"]]
    if TARGET_DOMAIN in domains:
        raise RuntimeError("Sketch cannot be a training or validation domain.")
    with open(resolve_repo_path(cfg["split_file"]), "r") as f:
        blob = json.load(f)
    root = resolve_data_root(cfg.get("data_root"))
    class_names = get_class_names()
    num_classes = int(cfg["num_classes"])
    out: Dict[str, Dict[str, List[Tuple[str, int]]]] = {}
    for d in domains:
        dd = find_domain_dir(root, d)
        out[d] = {}
        for part in ("train", "val"):
            entries = lookup_split(blob, d, part)
            if entries is None:
                # Debug output to help diagnose if it still misses
                keys_info = list(blob.keys()) if isinstance(blob, dict) else type(blob).__name__
                raise KeyError(
                    f"No '{part}' split for domain '{d}' in {cfg['split_file']}.\n"
                    f"Top-level keys found in your JSON: {keys_info}\n"
                    "If your JSON structure only saves indices instead of paths, `normalize_entry` will also need updating."
                )
            samples = []
            for e in entries:
                path, label = normalize_entry(e, class_names)
                if not 0 <= label < num_classes:
                    raise ValueError(f"Label {label} out of range for {num_classes} classes (domain {d}, {part}).")
                samples.append((resolve_sample_path(path, root, dd), label))
            out[d][part] = samples
    return out


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


class BalancedSourceLoader:
    """Loads balanced mini-batches with batch_per_domain examples per source domain."""

    def __init__(
        self,
        datasets: Mapping[str, Dataset],
        batch_per_domain: int,
        seed: int,
        num_workers: int = 0,
        epoch_mode: str = "max",
        pin_memory: bool = False,
    ) -> None:
        self.domains = list(datasets.keys())
        self.loaders: Dict[str, DataLoader] = {}
        for i, d in enumerate(self.domains):
            g = torch.Generator()
            g.manual_seed(seed + i)
            self.loaders[d] = DataLoader(
                datasets[d],
                batch_size=batch_per_domain,
                shuffle=True,
                drop_last=True,
                num_workers=num_workers,
                generator=g,
                worker_init_fn=_seed_worker,
                pin_memory=pin_memory,
                persistent_workers=num_workers > 0,
            )
        lengths = [len(l) for l in self.loaders.values()]
        if min(lengths) == 0:
            raise ValueError("A source train set has fewer examples than batch_per_domain.")
        self.steps_per_epoch = max(lengths) if epoch_mode == "max" else min(lengths)
        self._iters: Dict[str, Iterator] = {}

    def _next(self, d: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if d not in self._iters:
            self._iters[d] = iter(self.loaders[d])
        try:
            return next(self._iters[d])
        except StopIteration:
            self._iters[d] = iter(self.loaders[d])
            return next(self._iters[d])

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterator[Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        for _ in range(self.steps_per_epoch):
            xs, ys = [], []
            for d in self.domains:
                x, y = self._next(d)
                xs.append(x)
                ys.append(y)
            yield xs, ys


def make_val_loaders(
    val_datasets: Mapping[str, Dataset], batch_size: int = 64, num_workers: int = 0
) -> Dict[str, DataLoader]:
    return {
        d: DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        for d, ds in val_datasets.items()
    }


def balanced_val_subsets(
    val_datasets: Mapping[str, Dataset], n_per_domain: Optional[int] = None, seed: int = 6304
) -> Dict[str, Subset]:
    """Equal-sized random subsets of source validation sets."""
    domains = list(val_datasets.keys())
    n = min(len(val_datasets[d]) for d in domains) if n_per_domain is None else int(n_per_domain)
    rng = np.random.RandomState(seed)
    out: Dict[str, Subset] = {}
    for d in domains:
        if len(val_datasets[d]) < n:
            raise ValueError(f"Validation set of '{d}' has {len(val_datasets[d])} < {n} examples.")
        out[d] = Subset(val_datasets[d], rng.permutation(len(val_datasets[d]))[:n].tolist())
    return out


def stack_subsets(subsets: Mapping[str, Dataset]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack subsets into (x, y, domain_id) tensors."""
    xs, ys, ds = [], [], []
    for i, sub in enumerate(subsets.values()):
        for j in range(len(sub)):
            x, y = sub[j]
            xs.append(x)
            ys.append(y)
            ds.append(i)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long), torch.tensor(ds, dtype=torch.long)


def evaluate_source_val(
    model: nn.Module, val_loaders: Mapping[str, DataLoader], device: torch.device, num_classes: int
) -> Dict[str, Any]:
    return aggregate_domain_metrics(evaluate_domains(model, val_loaders, device, num_classes))


class SourceValidationSelector:
    """Tracks validation metrics, saves best checkpoint by mean macro-F1, and handles early stopping."""

    def __init__(self, run_dir: Path, patience: int = 5) -> None:
        self.run_dir = Path(run_dir)
        self.patience = patience
        self.ckpt_path = self.run_dir / "best.pt"
        self.best_score = -float("inf")
        self.best_epoch = 0
        self.best_metrics: Optional[Dict[str, Any]] = None
        self.epochs_since_best = 0
        self.history: List[Dict[str, Any]] = []

    def update(
        self, epoch: int, val_metrics: Dict[str, Any], model: nn.Module, config: Optional[Mapping[str, Any]] = None
    ) -> bool:
        score = float(val_metrics["mean_macro_f1"])
        improved = score > self.best_score + 1e-12
        self.history.append({"epoch": epoch, "selection_score": score, "improved": improved, "val": val_metrics})
        if improved:
            self.best_score, self.best_epoch, self.best_metrics = score, epoch, val_metrics
            self.epochs_since_best = 0
            torch.save(
                {
                    "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "epoch": epoch,
                    "selection_score": score,
                    "val_metrics": val_metrics,
                    "config": json.loads(json.dumps(config, default=str)) if config is not None else None,
                },
                self.ckpt_path,
            )
        else:
            self.epochs_since_best += 1
        with open(self.run_dir / "val_history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        return improved

    @property
    def should_stop(self) -> bool:
        return self.epochs_since_best >= self.patience