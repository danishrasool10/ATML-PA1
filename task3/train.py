#!/usr/bin/env python
"""Train ERM, DAN-DG, or SAM on PACS source domains."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from task3.evaluation.sharpness import sharpness_proxy
from task3.evaluation.source_domain_separability import source_domain_separability
from task3.methods.dan_dg import DANDGMethod
from task3.methods.erm import ERMMethod, load_task2_checkpoint
from task3.methods.sam import SAMMethod
from task3.models.backbone import assert_bn_frozen, bn_fingerprint, bn_unchanged, build_transforms, train_with_frozen_bn
from task3.models.classifier_head import DGModel
from task3.selection.source_validation import (
    BalancedSourceLoader,
    PACSDataset,
    SourceValidationSelector,
    balanced_val_subsets,
    evaluate_source_val,
    load_source_splits,
    make_val_loaders,
    resolve_repo_path,
    stack_subsets,
)

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
METHODS = {"erm": ERMMethod, "dan_dg": DANDGMethod, "sam": SAMMethod}
STUDY_PARAM = {"dan_dg": "lambda_dg", "sam": "rho"}
log = logging.getLogger("task3")


def deep_update(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(method: str, config_path: Optional[str], overrides: Dict[str, Any]) -> Dict[str, Any]:
    with open(CONFIG_DIR / "base.yaml") as f:
        cfg = yaml.safe_load(f)
    with open(config_path or CONFIG_DIR / f"{method}.yaml") as f:
        deep_update(cfg, yaml.safe_load(f))
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    cfg["method"] = method
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_file: Optional[Path] = None) -> None:
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    if log_file is not None:
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(fmt)
        log.addHandler(fh)


def run_name_for(cfg: Dict[str, Any], default_value: Optional[float]) -> str:
    method = cfg["method"]
    param = STUDY_PARAM.get(method)
    if param is None or default_value is None or float(cfg[param]) == float(default_value):
        return method
    return f"{method}_{param}{float(cfg[param]):g}"


def write_curves(rows: List[Dict[str, Any]], run_dir: Path) -> None:
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(run_dir / "curves.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ep = [r["epoch"] for r in rows]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(ep, [r["train_cls_loss"] for r in rows], marker="o", label="classification loss")
        if "train_perturbed_loss" in rows[0]:
            axes[0].plot(ep, [r["train_perturbed_loss"] for r in rows], marker="o", label="perturbed loss (SAM)")
        axes[0].set_title("Train classification loss")
        axes[0].legend()
        if "train_mmd" in rows[0]:
            axes[1].plot(ep, [r["train_mmd"] for r in rows], marker="o", color="tab:red")
        axes[1].set_title("Train MMD penalty (mean over pairs, unweighted)")
        axes[2].plot(ep, [r["val_mean_macro_f1"] for r in rows], marker="o", label="mean source F1")
        axes[2].plot(ep, [r["val_worst_macro_f1"] for r in rows], marker="o", label="worst source F1")
        axes[2].set_title("Source-validation macro-F1")
        axes[2].legend()
        for a in axes:
            a.set_xlabel("epoch")
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(run_dir / "curves.png", dpi=130)
        plt.close(fig)
    except Exception as exc:
        log.info(f"(curve plot skipped: {exc})")


def run_one(cfg: Dict[str, Any], device: torch.device, run_dir: Path) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "train.log")
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    method_name = cfg["method"]
    seed = int(cfg["seed"])
    seed_everything(seed)
    log.info(f"=== run '{run_dir.name}' | method={method_name} | device={device} | seed={seed} ===")

    domains = list(cfg["source_domains"])
    splits = load_source_splits(cfg)
    train_tf, eval_tf = build_transforms(cfg["resize"], cfg["crop"])
    train_ds = {d: PACSDataset(splits[d]["train"], train_tf) for d in splits}
    val_ds = {d: PACSDataset(splits[d]["val"], eval_tf) for d in splits}
    for d in splits:
        log.info(f"{d}: train={len(train_ds[d])} val={len(val_ds[d])}")
    num_classes = int(cfg["num_classes"])
    val_loaders = make_val_loaders(val_ds, cfg["val_batch_size"], cfg["num_workers"])
    train_loader = BalancedSourceLoader(
        train_ds, cfg["batch_per_domain"], seed, cfg["num_workers"], cfg["epoch_mode"], device.type == "cuda"
    )
    log.info(f"steps/epoch={len(train_loader)} (batch {cfg['batch_per_domain']} x {len(domains)} domains)")

    ckpt_path = cfg.get("load_from_task2") if method_name == "erm" else None
    model = DGModel(num_classes, pretrained=bool(cfg["pretrained"]) and not ckpt_path).to(device)
    if ckpt_path:
        how = load_task2_checkpoint(model, ckpt_path, device)
        log.info(f"Loaded Task 2 source-only checkpoint '{ckpt_path}' ({how}); ERM baseline is NOT retrained.")
    opt_cfg = cfg["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(opt_cfg["lr"]), weight_decay=float(opt_cfg["weight_decay"]))
    method = METHODS[method_name](model, optimizer, cfg)
    selector = SourceValidationSelector(run_dir, patience=int(cfg["patience"]))
    bn_before = bn_fingerprint(model)

    curves: List[Dict[str, Any]] = []
    if ckpt_path:
        val = evaluate_source_val(model, val_loaders, device, num_classes)
        selector.update(0, val, model, cfg)
        log.info(f"[Task2 ERM] val mean F1={val['mean_macro_f1']:.4f} worst F1={val['worst_macro_f1']:.4f}")
    else:
        for epoch in range(1, int(cfg["max_epochs"]) + 1):
            t0 = time.time()
            train_with_frozen_bn(model)  # BatchNorm in eval(): running stats frozen, affine params trainable
            assert_bn_frozen(model)
            sums: Dict[str, torch.Tensor] = {}
            steps = 0
            for xs, ys in train_loader:
                sizes = [x.size(0) for x in xs]
                x = torch.cat(xs).to(device, non_blocking=True)
                y = torch.cat(ys).to(device, non_blocking=True)
                out = method.train_step(x, y, sizes)
                for k, v in out.items():
                    sums[k] = sums.get(k, 0.0) + v
                steps += 1
            stats = {f"train_{k}": float(v) / steps for k, v in sums.items()}
            if not all(np.isfinite(list(stats.values()))):
                raise FloatingPointError(f"Non-finite training statistics at epoch {epoch}: {stats}")
            val = evaluate_source_val(model, val_loaders, device, num_classes)
            improved = selector.update(epoch, val, model, cfg)
            row: Dict[str, Any] = {"epoch": epoch, **stats}
            row.update(
                val_mean_acc=val["mean_acc"],
                val_mean_macro_f1=val["mean_macro_f1"],
                val_worst_acc=val["worst_acc"],
                val_worst_macro_f1=val["worst_macro_f1"],
            )
            for d, m in val["per_domain"].items():
                row[f"val_{d}_acc"] = m["acc"]
                row[f"val_{d}_macro_f1"] = m["macro_f1"]
            curves.append(row)
            log.info(
                f"ep {epoch:02d} | " + " ".join(f"{k[6:]}={v:.4f}" for k, v in stats.items())
                + f" | val meanF1={val['mean_macro_f1']:.4f} worstF1={val['worst_macro_f1']:.4f}"
                + f" meanAcc={val['mean_acc']:.4f} | {'*best*' if improved else f'no-improve {selector.epochs_since_best}/{selector.patience}'}"
                + f" | {time.time() - t0:.0f}s"
            )
            write_curves(curves, run_dir)
            if selector.should_stop:
                log.info(f"Early stopping at epoch {epoch} (best epoch {selector.best_epoch}).")
                break
        if not bn_unchanged(bn_before, bn_fingerprint(model)):
            raise RuntimeError("BatchNorm running statistics changed during training (policy violated).")
        log.info("Verified: BatchNorm running statistics unchanged during training.")

    try:
        best = torch.load(selector.ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        best = torch.load(selector.ckpt_path, map_location="cpu")
    model.load_state_dict(best["model"])
    model.to(device).eval()
    diag = cfg["diagnostics"]
    sep_subsets = balanced_val_subsets(val_ds, None, seed)
    separability = source_domain_separability(
        model, sep_subsets, device, seed, float(diag["separability_train_frac"]), cfg["val_batch_size"], cfg["num_workers"]
    )
    sharp_subsets = balanced_val_subsets(val_ds, int(diag["sharpness_per_domain"]), seed)
    xb, yb, _ = stack_subsets(sharp_subsets)
    sharp = sharpness_proxy(model, xb, yb, float(diag["sharpness_radius"]), device)
    log.info(f"Separability (chance {separability['chance']:.3f}): {separability['accuracy']:.4f}")
    log.info(f"Sharpness: L={sharp['loss']:.4f} L_eps={sharp['perturbed_loss']:.4f} delta={sharp['delta_sharp']:.4f}")

    summary = {
        "run_name": run_dir.name,
        "method": method_name,
        "config": json.loads(json.dumps(cfg, default=str)),
        "best_epoch": selector.best_epoch,
        "epochs_run": len(curves),
        "source_val": selector.best_metrics,
        "separability": separability,
        "sharpness": sharp,
        "checkpoint": str(selector.ckpt_path),
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(
        f"Best epoch {selector.best_epoch}: mean F1={selector.best_metrics['mean_macro_f1']:.4f} "
        f"worst F1={selector.best_metrics['worst_macro_f1']:.4f} -> {selector.ckpt_path}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 3 DG training (source domains only).")
    p.add_argument("--method", required=True, choices=list(METHODS))
    p.add_argument("--config", default=None, help="method YAML (default: task3/configs/<method>.yaml)")
    p.add_argument("--study", action="store_true", help="controlled study: lambda_DG for dan_dg, rho for sam")
    p.add_argument("--load_from_task2", default=None, help="ERM only: Task 2 source-only checkpoint")
    p.add_argument("--lambda_dg", type=float, default=None)
    p.add_argument("--rho", type=float, default=None)
    p.add_argument("--data_root", default=None)
    p.add_argument("--split_file", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--run_name", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--overwrite", action="store_true", help="re-run even if summary.json already exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.load_from_task2 and args.method != "erm":
        raise SystemExit("--load_from_task2 only applies to --method erm")
    overrides = {
        "data_root": args.data_root, "split_file": args.split_file, "output_dir": args.output_dir,
        "seed": args.seed, "max_epochs": args.max_epochs, "num_workers": args.num_workers,
        "load_from_task2": args.load_from_task2, "lambda_dg": args.lambda_dg, "rho": args.rho,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    cfg = load_config(args.method, args.config, {})
    default_value = cfg.get(STUDY_PARAM.get(args.method, ""), None)
    cfg = load_config(args.method, args.config, overrides)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_root = resolve_repo_path(cfg["output_dir"])
    setup_logging()

    def launch(c: Dict[str, Any]) -> Dict[str, Any]:
        name = args.run_name if (args.run_name and not args.study) else run_name_for(c, default_value)
        run_dir = out_root / name
        if (run_dir / "summary.json").exists() and not args.overwrite:
            log.info(f"Skipping '{name}': summary.json exists (use --overwrite to re-run).")
            with open(run_dir / "summary.json") as f:
                return json.load(f)
        return run_one(c, device, run_dir)

    if args.study:
        if args.method not in STUDY_PARAM:
            raise SystemExit("--study requires --method dan_dg (lambda_DG) or --method sam (rho).")
        param = STUDY_PARAM[args.method]
        rows = []
        for v in cfg["study_values"]:
            c = copy.deepcopy(cfg)
            c[param] = float(v)
            s = launch(c)
            sv = s["source_val"]
            rows.append({
                param: float(v), "run_name": s["run_name"], "best_epoch": s["best_epoch"],
                "val_mean_acc": sv["mean_acc"], "val_mean_macro_f1": sv["mean_macro_f1"],
                "val_worst_acc": sv["worst_acc"], "val_worst_macro_f1": sv["worst_macro_f1"],
                "separability": s["separability"]["accuracy"], "delta_sharp": s["sharpness"]["delta_sharp"],
            })
        study_dir = out_root / f"{args.method}_study"
        study_dir.mkdir(parents=True, exist_ok=True)
        with open(study_dir / "study_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        with open(study_dir / "study_summary.json", "w") as f:
            json.dump(rows, f, indent=2)
        setup_logging()
        for r in rows:
            log.info(json.dumps(r))
    else:
        launch(cfg)


if __name__ == "__main__":
    main()