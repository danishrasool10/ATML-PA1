"""Final Task 2 evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from shared.pacs import CLASS_NAMES, NUM_CLASSES, SOURCE_DOMAINS, resolve_pacs_root
from shared.pacs_protocol import (
    build_eval_loader,
    build_source_datasets,
    build_target_eval_dataset,
    deep_update,
    load_config,
    load_or_create_splits,
    parse_overrides,
    resolve_repo_path,
    set_seed,
)
from task2.evaluation.class_analysis import (
    compare_to_baseline,
    confusion_matrix_np,
    dominant_confusion,
    flip_examples,
    flip_statistics,
    per_class_accuracy,
    top_confusions,
)
from task2.evaluation.domain_separability import domain_separability_score
from task2.evaluation.metrics import classification_metrics, collect_outputs
from task2.models.classifier_head import build_classifier
from task2.train import CONFIG_DIR, METHOD_ORDER, STUDIES, study_runs

MAIN_COLUMNS = (
    ["method"]
    + [f"{d}_{k}" for d in SOURCE_DOMAINS for k in ("acc", "f1")]
    + ["mean_src_acc", "mean_src_f1", "target_acc", "target_f1", "delta_target_acc_pp",
       "gap_src_minus_tgt_pp", "domain_separability", "best_epoch"]
)

def pct(x) -> float:
    return round(100.0 * float(x), 2)

def load_run(run_dir: Path, device):
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model = build_classifier(NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt

def evaluate_run(run_dir: Path, val_loaders, tgt_loader, device, seed: int, standardize: bool) -> dict:
    model, ckpt = load_run(run_dir, device)
    per_domain, src_feats = {}, []
    for domain, loader in val_loaders.items():
        out = collect_outputs(model, loader, device, want_features=True)
        per_domain[domain] = classification_metrics(out["labels"], out["preds"], NUM_CLASSES)
        src_feats.append(out["features"])
    
    tgt = collect_outputs(model, tgt_loader, device, want_features=True)
    tgt_m = classification_metrics(tgt["labels"], tgt["preds"], NUM_CLASSES)
    sep = domain_separability_score(np.concatenate(src_feats), tgt["features"], seed=seed, standardize=standardize)

    mean_f1 = float(np.mean([m["macro_f1"] for m in per_domain.values()]))
    recorded = (ckpt.get("val") or {}).get("mean_macro_f1")
    if recorded is not None and abs(recorded - mean_f1) > 1e-4:
        print(f"[warn] {run_dir.name}: recomputed source-val macro-F1 {mean_f1:.4f} != training-time {recorded:.4f}")
    
    return {
        "run": run_dir.name,
        "best_epoch": int(ckpt["epoch"]),
        "source": per_domain,
        "mean_src_acc": float(np.mean([m["acc"] for m in per_domain.values()])),
        "mean_src_f1": mean_f1,
        "target_acc": tgt_m["acc"],
        "target_f1": tgt_m["macro_f1"],
        "separability": sep["accuracy"],
        "sep_details": sep,
        "target_preds": tgt["preds"],
        "target_labels": tgt["labels"],
    }

def write_table(rows: List[dict], columns: List[str], stem: Path) -> str:
    with open(stem.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    lines += ["| " + " | ".join(str(r[c]) for c in columns) + " |" for r in rows]
    text = "\n".join(lines) + "\n"
    stem.with_suffix(".md").write_text(text)
    return text

def main_row(name: str, res: dict, base: dict) -> dict:
    row = {"method": name}
    for d in SOURCE_DOMAINS:
        row[f"{d}_acc"] = pct(res["source"][d]["acc"])
        row[f"{d}_f1"] = pct(res["source"][d]["macro_f1"])
    row.update(
        mean_src_acc=pct(res["mean_src_acc"]), mean_src_f1=pct(res["mean_src_f1"]),
        target_acc=pct(res["target_acc"]), target_f1=pct(res["target_f1"]),
        delta_target_acc_pp=pct(res["target_acc"] - base["target_acc"]),
        gap_src_minus_tgt_pp=pct(res["mean_src_acc"] - res["target_acc"]),
        domain_separability=pct(res["separability"]), best_epoch=res["best_epoch"],
    )
    return row

def strip_arrays(res: dict) -> dict:
    return {k: v for k, v in res.items() if k not in ("target_preds", "target_labels")}

def fmt_conf(c) -> str:
    return "none" if c is None else f"{c['true']}→{c['pred']} ({100 * c['rate']:.1f}%, n={c['count']})"

def run_class_analysis(results: Dict[str, dict], tgt_samples, out_dir: Path) -> dict:
    labels = results["source_only"]["target_labels"]
    base_preds = results["source_only"]["target_preds"]
    cms = {m: confusion_matrix_np(r["target_labels"], r["target_preds"], NUM_CLASSES) for m, r in results.items()}
    base_cm = cms["source_only"]
    base_acc = per_class_accuracy(base_cm)

    report = {
        "source_only": {
            "per_class_accuracy": {n: float(base_acc[i]) for i, n in enumerate(CLASS_NAMES)},
            "top_confusions": top_confusions(base_cm, CLASS_NAMES, 6),
        },
        "methods": {},
    }
    
    lines = ["# Target class-level analysis", "", "## Source-only failures (worst class first)", ""]
    for i in np.argsort(base_acc):
        lines.append(f"- {CLASS_NAMES[i]}: {100 * base_acc[i]:.1f}% | dominant confusion: "
                     f"{fmt_conf(dominant_confusion(base_cm, int(i), CLASS_NAMES))}")
    lines += ["", "Top confusions: " + "; ".join(fmt_conf(c) for c in report["source_only"]["top_confusions"]), ""]

    for m, res in results.items():
        if m == "source_only":
            continue
        cmp_ = compare_to_baseline(base_cm, cms[m], CLASS_NAMES)
        flips = flip_statistics(labels, base_preds, res["target_preds"], CLASS_NAMES)
        worst = cmp_["largest_degradation"]["class"]
        harmed = flip_examples(labels, base_preds, res["target_preds"], CLASS_NAMES.index(worst), "harmed", 5)
        
        report["methods"][m] = {
            "comparison": cmp_, "flips": flips,
            "harmed_examples_in_worst_class": [tgt_samples[i][0] for i in harmed],
        }
        
        up, down = cmp_["largest_improvement"], cmp_["largest_degradation"]
        lines += [
            f"## {m}",
            f"- target acc change: {pct(res['target_acc'] - results['source_only']['target_acc']):+.2f} pp | "
            f"helped {flips['helped']} vs harmed {flips['harmed']} samples | classes degraded: {cmp_['num_classes_degraded']}",
            f"- largest improvement: {up['class']} ({up['delta_pp']:+.1f} pp); dominant confusion "
            f"{fmt_conf(up['baseline_dominant_confusion'])} -> {fmt_conf(up['method_dominant_confusion'])}",
            f"- largest degradation: {down['class']} ({down['delta_pp']:+.1f} pp, "
            f"negative transfer={down['is_negative_transfer']}); dominant confusion "
            f"{fmt_conf(down['baseline_dominant_confusion'])} -> {fmt_conf(down['method_dominant_confusion'])}",
            f"- example harmed samples ({worst}): {', '.join(report['methods'][m]['harmed_examples_in_worst_class']) or 'none'}",
            "",
        ]

    (out_dir / "class_analysis.md").write_text("\n".join(lines))
    (out_dir / "class_analysis.json").write_text(json.dumps(report, indent=2))
    (out_dir / "confusion_matrices.json").write_text(
        json.dumps({"class_names": CLASS_NAMES, **{m: cm.tolist() for m, cm in cms.items()}}, indent=2))

    with open(out_dir / "per_class_target_accuracy.csv", "w", newline="") as f:
        w = csv.writer(f)
        header = ["class", "n"] + [f"{m}_acc" for m in results] + [f"{m}_delta_pp" for m in results if m != "source_only"]
        w.writerow(header)
        for i, name in enumerate(CLASS_NAMES):
            row = [name, int(base_cm[i].sum())]
            row += [round(100 * float(per_class_accuracy(cms[m])[i]), 2) for m in results]
            row += [round(100 * float(per_class_accuracy(cms[m])[i] - base_acc[i]), 2) for m in results if m != "source_only"]
            w.writerow(row)
    return report

def plot_class_deltas(report: dict, out_path: Path) -> None:
    methods = list(report["methods"])
    if not methods:
        return
    x, width = np.arange(len(CLASS_NAMES)), 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, m in enumerate(methods):
        deltas = [c["delta_pp"] for c in report["methods"][m]["comparison"]["per_class"]]
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, deltas, width, label=m)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("Δ target class accuracy vs Source-only (pp)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_training_curves(results_root: Path, methods: List[str], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for m in methods:
        hist_path = results_root / m / "history.json"
        if not hist_path.exists():
            continue
        hist = json.loads(hist_path.read_text())
        ep = [h["epoch"] for h in hist]
        
        axes[0, 0].plot(ep, [h["train"]["cls"] for h in hist], label=m)
        if hist[0]["train"].get("align") is not None:
            axes[0, 1].plot(ep, [h["train"]["align"] for h in hist], label=m)
        if hist[0]["train"].get("domain_acc") is not None:
            axes[1, 0].plot(ep, [h["train"]["domain_acc"] for h in hist], label=m)
        axes[1, 1].plot(ep, [h["val"]["mean_macro_f1"] for h in hist], label=m)
        
    axes[0, 0].set_title("Source classification loss (train)")
    axes[0, 1].set_title("Alignment loss (DAN: MMD², DANN/CDAN: domain CE)")
    axes[1, 0].set_title("Domain-discriminator accuracy (dotted = chance)")
    axes[1, 0].axhline(0.5, ls=":", color="gray")
    axes[1, 1].set_title("Mean source-validation macro-F1")
    
    for ax in axes.ravel():
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def run_design_study(method, overrides, results_root, val_loaders, tgt_loader, device, seed, standardize, out_dir) -> None:
    study = STUDIES[method]
    rows = []
    for value, run_name, _ in study_runs(method, overrides):
        run_dir = results_root / run_name
        if not (run_dir / "best.pt").exists():
            sys.exit(f"Missing run '{run_name}'. Train it first: python -m task2.train --study {method}")
        res = evaluate_run(run_dir, val_loaders, tgt_loader, device, seed, standardize)
        rows.append({
            study["param"]: value, "run": run_name, "best_epoch": res["best_epoch"],
            "mean_src_acc": pct(res["mean_src_acc"]), "mean_src_f1": pct(res["mean_src_f1"]),
            "target_acc": pct(res["target_acc"]), "target_f1": pct(res["target_f1"]),
            "domain_separability": pct(res["separability"]),
        })
        
    columns = [study["param"], "run", "best_epoch", "mean_src_acc", "mean_src_f1", "target_acc", "target_f1", "domain_separability"]
    print(f"\nDesign study ({method}) — analysis only; do NOT use these target results to revise settings.\n")
    print(write_table(rows, columns, out_dir / f"design_study_{method}"))

    xs = [r[study["param"]] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for key, label in [("mean_src_f1", "source-val macro-F1"), ("target_acc", "target accuracy"),
                       ("domain_separability", "domain separability")]:
        ax.plot(xs, [r[key] for r in rows], marker="o", label=label)
        
    ax.axhline(50, ls=":", color="gray", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.minorticks_off()
    ax.set_xlabel(study["param"])
    ax.set_ylabel("%")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"design_study_{method}.png", dpi=150)
    plt.close(fig)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Final Task 2 evaluation.")
    p.add_argument("--data_root", default=None)
    p.add_argument("--methods", nargs="+", default=METHOD_ORDER)
    p.add_argument("--design_study", choices=sorted(STUDIES), default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--standardize_features", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args()

def main() -> None:
    args = parse_args()
    overrides = parse_overrides(args.overrides)
    if args.data_root:
        overrides = deep_update(overrides, {"data": {"root": args.data_root}})
        
    cfg = load_config(CONFIG_DIR / "base.yaml", overrides)
    seed = int(cfg["seed"])
    set_seed(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    results_root = resolve_repo_path(cfg["output"]["dir"])
    out_dir = Path(args.out_dir) if args.out_dir else results_root / "final"
    out_dir.mkdir(parents=True, exist_ok=True)

    if "source_only" not in args.methods:
        sys.exit("'source_only' is required.")
        
    missing = [m for m in args.methods if not (results_root / m / "best.pt").exists()]
    if missing:
        sys.exit(f"Missing trained runs {missing}.")

    root = resolve_pacs_root(resolve_repo_path(cfg["data"]["root"]))
    splits = load_or_create_splits(root, resolve_repo_path(cfg["data"]["split_file"]), seed=seed)
    size, crop = int(cfg["data"]["image_size"]), int(cfg["data"]["crop_size"])
    bs, workers = int(cfg["data"]["eval_batch_size"]), int(cfg["data"]["num_workers"])
    
    _, src_val = build_source_datasets(root, splits, size, crop)
    tgt_ds = build_target_eval_dataset(root, splits, size, crop)
    val_loaders = {d: build_eval_loader(ds, bs, workers) for d, ds in src_val.items()}
    tgt_loader = build_eval_loader(tgt_ds, bs, workers)

    results: Dict[str, dict] = {}
    for m in args.methods:
        print(f"[eval] {m}")
        results[m] = evaluate_run(results_root / m, val_loaders, tgt_loader, device, seed, args.standardize_features)
        
        with open(out_dir / f"target_predictions_{m}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "pred"])
            for (rel, _), y, p in zip(tgt_ds.samples, results[m]["target_labels"], results[m]["target_preds"]):
                w.writerow([rel, CLASS_NAMES[y], CLASS_NAMES[p]])

    base = results["source_only"]
    rows = [main_row(m, r, base) for m, r in results.items()]
    print("\n" + write_table(rows, MAIN_COLUMNS, out_dir / "main_results"))
    (out_dir / "main_results.json").write_text(json.dumps({m: strip_arrays(r) for m, r in results.items()}, indent=2))
    print(f"Source-only domain gap: {pct(base['mean_src_acc'] - base['target_acc']):.2f} pp")

    report = run_class_analysis(results, tgt_ds.samples, out_dir)
    plot_class_deltas(report, out_dir / "per_class_delta.png")
    print((out_dir / "class_analysis.md").read_text())

    plot_training_curves(results_root, args.methods, out_dir / "training_curves.png")

    if args.design_study:
        run_design_study(args.design_study, overrides, results_root, val_loaders, tgt_loader, device, seed,
                         args.standardize_features, out_dir)

    print(f"\nAll outputs written to {out_dir}")

if __name__ == "__main__":
    main()