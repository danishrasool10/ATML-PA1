"""Task 2 training entry point: Source-only ERM, DAN, DANN, CDAN, and controlled design study."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from tqdm import tqdm

from shared.pacs import NUM_CLASSES, resolve_pacs_root
from shared.pacs_protocol import (
    build_eval_loader,
    build_source_datasets,
    build_source_loader,
    build_target_adaptation_dataset,
    build_target_loader,
    cycle_loader,
    deep_update,
    default_steps_per_epoch,
    load_config,
    load_or_create_splits,
    parse_overrides,
    resolve_repo_path,
    set_seed,
)
from task2.evaluation.metrics import evaluate_source_validation
from task2.methods.cdan import CDANMethod
from task2.methods.dan import DANMethod, STUDY as DAN_STUDY
from task2.methods.dann import DANNMethod, STUDY as DANN_STUDY
from task2.methods.source_only import SourceOnlyMethod
from task2.models.backbone import freeze_batchnorm_statistics, snapshot_bn_buffers, verify_bn_buffers_unchanged
from task2.models.classifier_head import build_classifier

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
METHODS = {"source_only": SourceOnlyMethod, "dan": DANMethod, "dann": DANNMethod, "cdan": CDANMethod}
METHOD_ORDER = ["source_only", "dan", "dann", "cdan"]
STUDIES = {"dan": DAN_STUDY, "dann": DANN_STUDY}


def study_run_name(method: str, study: dict, value: float, base_value: float) -> str:
    if abs(float(value) - float(base_value)) < 1e-12:
        return method
    return f"{method}_{study['tag']}{float(value):g}"


def study_runs(method: str, overrides: Optional[dict] = None) -> List[Tuple[float, str, dict]]:
    study = STUDIES[method]
    base_cfg = load_config(CONFIG_DIR / f"{method}.yaml", overrides)
    base_value = float(base_cfg["method"][study["param"]])
    runs = []
    for value in study["values"]:
        cfg = deep_update(base_cfg, {"method": {study["param"]: value}})
        runs.append((float(value), study_run_name(method, study, value, base_value), cfg))
    return runs


def _fmt(v: Optional[float]) -> str:
    return "  n/a " if v is None else f"{v:.4f}"


def train_one_run(cfg: dict, run_name: str, force: bool = False, device_name: Optional[str] = None) -> Path:
    results_root = resolve_repo_path(cfg["output"]["dir"])
    out_dir = results_root / run_name
    if (out_dir / "best.pt").exists() and not force:
        print(f"[skip] '{run_name}' already trained ({out_dir}); pass --force to retrain.")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg["seed"])
    set_seed(seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    d_cfg, t_cfg, m_cfg = cfg["data"], cfg["train"], cfg["method"]
    
    if m_cfg["name"] not in METHODS:
        raise ValueError(f"Unknown method '{m_cfg['name']}'")

    root = resolve_pacs_root(resolve_repo_path(d_cfg["root"]))
    splits = load_or_create_splits(root, resolve_repo_path(d_cfg["split_file"]), seed=seed)
    image_size, crop_size, workers = int(d_cfg["image_size"]), int(d_cfg["crop_size"]), int(d_cfg["num_workers"])
    src_train, src_val = build_source_datasets(root, splits, image_size, crop_size)

    per_domain = int(t_cfg["source_per_domain_batch"])
    steps_per_epoch = int(t_cfg.get("steps_per_epoch") or default_steps_per_epoch(src_train, per_domain))
    max_epochs, patience = int(t_cfg["max_epochs"]), int(t_cfg["patience"])
    total_steps = max_epochs * steps_per_epoch

    src_loader = build_source_loader(src_train, per_domain, steps_per_epoch, seed, workers)
    val_loaders = {d: build_eval_loader(ds, int(d_cfg["eval_batch_size"]), workers) for d, ds in src_val.items()}

    model = build_classifier(int(cfg["model"]["num_classes"]), bool(cfg["model"]["pretrained"])).to(device)
    method = METHODS[m_cfg["name"]](m_cfg, model).to(device)
    bn_reference = snapshot_bn_buffers(model)
    optimizer = torch.optim.AdamW(
        itertools.chain(model.parameters(), method.parameters()),
        lr=float(t_cfg["lr"]), weight_decay=float(t_cfg["weight_decay"]),
    )

    tgt_iter = None
    if method.uses_target:
        tgt_ds = build_target_adaptation_dataset(root, splits, image_size, crop_size)
        tgt_iter = cycle_loader(build_target_loader(tgt_ds, int(t_cfg["target_batch"]), seed, workers))

    print(f"\n=== {run_name} | method={m_cfg['name']} | device={device} | steps/epoch={steps_per_epoch} "
          f"| source-train={sum(len(d) for d in src_train.values())} ===")

    history, best_f1, best_epoch, best_val, stale, global_step = [], -1.0, 0, None, 0, 0
    start = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        freeze_batchnorm_statistics(model)
        method.train()
        agg = defaultdict(list)
        t0 = time.time()

        pbar = tqdm(src_loader, total=steps_per_epoch, desc=f"[{run_name}] ep {epoch}/{max_epochs}", leave=False)
        for src_x, src_y, _ in pbar:
            src_x, src_y = src_x.to(device, non_blocking=True), src_y.to(device, non_blocking=True)
            tgt_x = None
            if tgt_iter is not None:
                tgt_x, tgt_y, _ = next(tgt_iter)
                assert bool((tgt_y < 0).all()), "Target labels must not reach the training loop."
                tgt_x = tgt_x.to(device, non_blocking=True)

            out = method.compute_loss(model, src_x, src_y, tgt_x, progress=global_step / total_steps)
            if not torch.isfinite(out.total):
                raise FloatingPointError(f"Non-finite loss at step {global_step}: {out.logs}")
            
            optimizer.zero_grad(set_to_none=True)
            out.total.backward()
            optimizer.step()

            for k, v in out.logs.items():
                agg[k].append(v)
            global_step += 1
            pbar.set_postfix(loss=f"{out.logs['total']:.3f}")

        train_log = {
            k: (float(np.mean([x for x in v if x is not None])) if any(x is not None for x in v) else None)
            for k, v in agg.items()
        }

        val = evaluate_source_validation(model, val_loaders, device, NUM_CLASSES)
        verify_bn_buffers_unchanged(model, bn_reference)

        improved = val["mean_macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch, best_val, stale = val["mean_macro_f1"], epoch, val, 0
            torch.save({"model": model.state_dict(), "method": method.state_dict(), "epoch": epoch,
                        "val": val, "cfg": cfg, "run_name": run_name}, out_dir / "best.pt")
        else:
            stale += 1

        history.append({"epoch": epoch, "train": train_log, "val": val, "seconds": time.time() - t0})
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        
        print(f"[{run_name}] ep {epoch:02d} | cls {_fmt(train_log['cls'])} | align {_fmt(train_log['align'])} "
              f"| dom-acc {_fmt(train_log['domain_acc'])} | val acc {val['mean_acc']:.4f} "
              f"| val macro-F1 {val['mean_macro_f1']:.4f} | best {best_f1:.4f}@{best_epoch}{' *' if improved else ''}")
        
        if stale >= patience:
            print(f"[{run_name}] early stopping: no improvement for {patience} epochs.")
            break

    summary = {
        "run_name": run_name, "method": m_cfg["name"], "best_epoch": best_epoch, "epochs_run": len(history),
        "best_mean_source_val_macro_f1": best_f1, "source_val_at_best": best_val,
        "steps_per_epoch": steps_per_epoch, "seconds": time.time() - start,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "config.json").write_text(json.dumps({**cfg, "run_name": run_name}, indent=2))
    print(f"[{run_name}] done -> {out_dir}")

    del tgt_iter, src_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 2 UDA training (PACS -> Sketch).")
    p.add_argument("--config", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--study", choices=sorted(STUDIES), default=None)
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--run_name", default=None)
    p.add_argument("--data_root", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    overrides = parse_overrides(args.overrides)
    if args.data_root:
        overrides = deep_update(overrides, {"data": {"root": args.data_root}})

    jobs: List[Tuple[dict, str]] = []
    if args.all:
        for name in METHOD_ORDER:
            cfg = load_config(CONFIG_DIR / f"{name}.yaml", overrides)
            jobs.append((cfg, cfg["output"].get("run_name") or name))
    if args.study:
        jobs += [(cfg, run) for _, run, cfg in study_runs(args.study, overrides)]
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            cfg_path = REPO_ROOT / args.config
        cfg = load_config(cfg_path, overrides)
        jobs.append((cfg, args.run_name or cfg["output"].get("run_name") or cfg["method"]["name"]))
        
    if not jobs:
        raise SystemExit("Nothing to do: pass --all, --study {dan,dann} and/or --config <yaml>.")

    seen = set()
    for cfg, run_name in jobs:
        if run_name in seen:
            continue
        seen.add(run_name)
        train_one_run(cfg, run_name, force=args.force, device_name=args.device)


if __name__ == "__main__":
    main()