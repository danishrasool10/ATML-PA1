#!/usr/bin/env python
"""Evaluate frozen Task 3 checkpoints on the Sketch domain."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from task3.evaluation.domain_metrics import compute_metrics, confusion, per_class_accuracy, predict
from task3.models.backbone import build_transforms
from task3.models.classifier_head import load_dg_checkpoint
from task3.selection.source_validation import (
    IMG_EXTS,
    TARGET_DOMAIN,
    PACSDataset,
    find_domain_dir,
    get_class_names,
    lookup_domain_all,
    normalize_entry,
    resolve_data_root,
    resolve_repo_path,
    resolve_sample_path,
)

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
MAIN_RUNS = ["erm", "dan_dg", "sam"]


def load_sketch_samples(cfg: Dict[str, Any], class_names: List[str]) -> List[tuple]:
    root = resolve_data_root(cfg.get("data_root"))
    dd = find_domain_dir(root, TARGET_DOMAIN)
    samples: List[tuple] = []
    if dd is not None:
        name2idx = {n: i for i, n in enumerate(class_names)}
        for cdir in sorted(p for p in dd.iterdir() if p.is_dir()):
            if cdir.name.lower() not in name2idx:
                continue
            for f in sorted(cdir.iterdir()):
                if f.suffix.lower() in IMG_EXTS:
                    samples.append((str(f), name2idx[cdir.name.lower()]))
    if samples:
        return samples
    with open(resolve_repo_path(cfg["split_file"])) as f:
        entries = lookup_domain_all(json.load(f), TARGET_DOMAIN)
    if entries is None:
        raise RuntimeError("Could not locate Sketch images: set --data_root (expects <root>/sketch/<class>/*.jpg).")
    for e in entries:
        path, label = normalize_entry(e, class_names)
        samples.append((resolve_sample_path(path, root, dd), label))
    return samples


def print_table(headers: List[str], rows: List[List[str]], title: str = "") -> None:
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    if title:
        print(f"\n{title}")
    print(" | ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def pct(x: float) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{100 * x:.2f}"


def save_confusion_plot(cms: Dict[str, np.ndarray], class_names: List[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.8), squeeze=False)
    for ax, (name, cm) in zip(axes[0], cms.items()):
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
        ax.set_title(f"{name} (row-normalized)")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate frozen Task 3 checkpoints on Sketch.")
    ap.add_argument("--runs", nargs="+", default=MAIN_RUNS, help="run directory names under --results_dir")
    ap.add_argument("--include_study", action="store_true", help="also evaluate dan_dg_lambda_dg* and sam_rho* runs")
    ap.add_argument("--results_dir", default=None)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--split_file", default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    with open(CONFIG_DIR / "base.yaml") as f:
        cfg = yaml.safe_load(f)
    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.split_file:
        cfg["split_file"] = args.split_file
    results_dir = resolve_repo_path(args.results_dir or cfg["output_dir"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    num_classes = int(cfg["num_classes"])
    class_names = get_class_names()

    names = list(args.runs)
    if args.include_study:
        for pat in ("dan_dg_lambda_dg*", "sam_rho*"):
            for p in sorted(glob.glob(str(results_dir / pat))):
                if Path(p).name not in names and (Path(p) / "best.pt").exists():
                    names.append(Path(p).name)

    _, eval_tf = build_transforms(cfg["resize"], cfg["crop"])
    samples = load_sketch_samples(cfg, class_names)
    sketch_ds = PACSDataset(samples, eval_tf)
    loader = DataLoader(sketch_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Sketch images: {len(sketch_ds)} | device: {device}")

    res: Dict[str, Dict[str, Any]] = {}
    out_dir = results_dir / "sketch_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        run_dir = results_dir / name
        with open(run_dir / "summary.json") as f:
            summ = json.load(f)
        model = load_dg_checkpoint(str(run_dir / "best.pt"), num_classes, device)
        y_true, y_pred = predict(model, loader, device)
        m = compute_metrics(y_true, y_pred, num_classes)
        res[name] = {
            "summary": summ, "sketch": m,
            "per_class_acc": per_class_accuracy(y_true, y_pred, num_classes),
            "cm": confusion(y_true, y_pred, num_classes),
        }
        with open(out_dir / f"predictions_{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "true", "pred"])
            for (path, _), t, p in zip(sketch_ds.samples, y_true, y_pred):
                w.writerow([path, class_names[t], class_names[p]])
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    base_acc = res["erm"]["sketch"]["acc"] if "erm" in res else float("nan")
    base_pc = res["erm"]["per_class_acc"] if "erm" in res else np.full(num_classes, np.nan)
    doms = list(next(iter(res.values()))["summary"]["source_val"]["per_domain"].keys())

    headers = ["method"] + [f"{d} acc/F1" for d in doms] + ["mean acc/F1", "worst acc/F1", "Sketch acc", "Sketch F1", "dAcc vs ERM (pp)"]
    rows, csv_rows = [], []
    for name, r in res.items():
        sv = r["summary"]["source_val"]
        cells = [f"{pct(sv['per_domain'][d]['acc'])}/{pct(sv['per_domain'][d]['macro_f1'])}" for d in doms]
        delta = 100 * (r["sketch"]["acc"] - base_acc)
        rows.append([name] + cells + [
            f"{pct(sv['mean_acc'])}/{pct(sv['mean_macro_f1'])}", f"{pct(sv['worst_acc'])}/{pct(sv['worst_macro_f1'])}",
            pct(r["sketch"]["acc"]), pct(r["sketch"]["macro_f1"]), f"{delta:+.2f}",
        ])
        csv_rows.append(dict(
            method=name,
            **{f"{d}_acc": sv["per_domain"][d]["acc"] for d in doms},
            **{f"{d}_macro_f1": sv["per_domain"][d]["macro_f1"] for d in doms},
            mean_acc=sv["mean_acc"], mean_macro_f1=sv["mean_macro_f1"],
            worst_acc=sv["worst_acc"], worst_macro_f1=sv["worst_macro_f1"],
            sketch_acc=r["sketch"]["acc"], sketch_macro_f1=r["sketch"]["macro_f1"],
            delta_sketch_acc_vs_erm=r["sketch"]["acc"] - base_acc,
            separability=r["summary"]["separability"]["accuracy"], delta_sharp=r["summary"]["sharpness"]["delta_sharp"],
        ))
    print_table(headers, rows, "Source validation (%) and Sketch performance (%)")
    with open(out_dir / "main_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    rows = []
    for name, r in res.items():
        s, sh = r["summary"]["separability"], r["summary"]["sharpness"]
        rows.append([name, f"{pct(s['accuracy'])} (chance {pct(s['chance'])})", f"{sh['loss']:.4f}",
                     f"{sh['perturbed_loss']:.4f}", f"{sh['delta_sharp']:.4f}"])
    print_table(["method", "source separability (%)", "L_val", "L_val(theta+eps)", "Delta_sharp"], rows,
                "Source-domain separability and common sharpness proxy (source validation data)")

    headers = ["class"] + [h for n in res for h in ([n] if n == "erm" else [n, f"{n} d"])]
    rows, pc_csv = [], []
    for c, cn in enumerate(class_names):
        row, crow = [cn], {"class": cn}
        for n, r in res.items():
            a = r["per_class_acc"][c]
            crow[f"{n}_acc"] = float(a)
            if n == "erm":
                row.append(pct(a))
            else:
                d = 100 * (a - base_pc[c])
                row += [pct(a), f"{d:+.2f}"]
                crow[f"{n}_delta_vs_erm"] = float(a - base_pc[c])
        rows.append(row)
        pc_csv.append(crow)
    print_table(headers, rows, "Per-class Sketch accuracy (%) and change vs ERM (pp)")
    with open(out_dir / "per_class_sketch.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pc_csv[0].keys()))
        w.writeheader()
        w.writerows(pc_csv)

    print("\nPositive / negative transfer vs ERM (top 3 each, pp) and top-5 confusions (true -> pred, share of true class)")
    analysis: Dict[str, Any] = {}
    for n, r in res.items():
        cm = r["cm"]
        off = [(cm[i, j], i, j) for i in range(num_classes) for j in range(num_classes) if i != j]
        off.sort(reverse=True)
        conf = [(class_names[i], class_names[j], int(k), float(k / max(cm[i].sum(), 1))) for k, i, j in off[:5]]
        entry: Dict[str, Any] = {"top_confusions": conf}
        if n != "erm" and "erm" in res:
            dlt = r["per_class_acc"] - base_pc
            order = [i for i in np.argsort(dlt) if np.isfinite(dlt[i])]
            entry["improvements"] = [(class_names[i], float(dlt[i])) for i in order[::-1][:3] if dlt[i] > 0]
            entry["degradations"] = [(class_names[i], float(dlt[i])) for i in order[:3] if dlt[i] < 0]
            print(f"[{n}] gains: " + (", ".join(f"{c} {100 * d:+.1f}" for c, d in entry["improvements"]) or "none")
                  + " | losses: " + (", ".join(f"{c} {100 * d:+.1f}" for c, d in entry["degradations"]) or "none"))
        print(f"[{n}] confusions: " + "; ".join(f"{a}->{b} {k} ({100 * s:.1f}%)" for a, b, k, s in conf))
        analysis[n] = entry
        np.savetxt(out_dir / f"confusion_{n}.csv", cm, fmt="%d", delimiter=",",
                   header=",".join(class_names), comments="")
    save_confusion_plot({n: r["cm"] for n, r in res.items()}, class_names, out_dir / "confusion_matrices.png")

    with open(out_dir / "sketch_results.json", "w") as f:
        json.dump({
            "n_sketch": len(sketch_ds),
            "class_names": class_names,
            "runs": {n: {
                "run_name": n,
                "source_val": r["summary"]["source_val"],
                "separability": r["summary"]["separability"],
                "sharpness": r["summary"]["sharpness"],
                "sketch": r["sketch"],
                "delta_sketch_acc_vs_erm": r["sketch"]["acc"] - base_acc,
                "per_class_acc": r["per_class_acc"].tolist(),
                "confusion_matrix": r["cm"].tolist(),
                "analysis": analysis[n],
            } for n, r in res.items()},
        }, f, indent=2)
    print(f"\nSaved tables, confusion matrices, and predictions to {out_dir}")


if __name__ == "__main__":
    main()