"""Evaluate OSR scores and methods from cached outputs."""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from data.cifar10 import CIFAR10_CLASSES
from data.cifar100_unknowns import GROUP_FAR, GROUP_NEAR, load_unknowns
from evaluation.failure_analysis import (collect_failures, format_failures, per_class_acceptance,
                                         save_failure_images)
from evaluation.metrics import closed_set_accuracy, osr_metrics
from extract_outputs import (METHODS, ckpt_path, ensure_known_cache, ensure_unknown_cache, load_cache)
from methods.proser import placeholder_unknownness
from scores.energy import energy_score
from scores.mahalanobis import fit_mahalanobis
from scores.mls import mls_score
from scores.msp import msp_score

ROOT = Path(__file__).resolve().parent
NUM_KNOWN = 10

COLUMNS = [("auroc_near", "AUROC Near"), ("auroc_far", "AUROC Far"), ("auroc_all", "AUROC All"),
           ("fpr_near", "FPR@95TPR Near"), ("fpr_far", "FPR@95TPR Far"), ("fpr_all", "FPR@95TPR All"),
           ("known_accept", "Known Accept (test)"), ("reject_near", "Near Reject"), ("reject_far", "Far Reject")]
PCT_KEYS = {"fpr_near", "fpr_far", "fpr_all", "known_accept", "reject_near", "reject_far", "csa"}


def score_splits(score_fn, cache, num_logits=None):
    """Apply a score to val/test/unknown using the same saved logits/features."""
    out = {}
    for s in ("val", "test", "unknown"):
        lg = cache[s]["logits"]
        if num_logits is not None:
            lg = lg[:, :num_logits]
        out[s] = score_fn(lg, cache[s]["features"])
    return out


def make_row(label_cols, sc, cache, csa=None):
    g = cache["unknown"]["group"]
    m = osr_metrics(sc["val"], sc["test"], sc["unknown"][g == GROUP_NEAR], sc["unknown"][g == GROUP_FAR])
    row = dict(label_cols)
    if csa is not None:
        row["csa"] = csa
    row.update(m)
    return row


def fmt(key, v):
    if key in PCT_KEYS:
        return f"{100 * v:.2f}%"
    return f"{v:.4f}"


def write_table(rows, label_keys, stem, include_csa=False):
    cols = [(k, k.capitalize()) for k in label_keys] + ([("csa", "CSA")] if include_csa else []) + COLUMNS
    cols.append(("tau", "Threshold"))
    with open(f"{stem}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([h for _, h in cols])
        for r in rows:
            w.writerow([r[k] for k, _ in cols])
    lines = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = [str(r[k]) if k in label_keys else fmt(k, r[k]) for k, _ in cols]
        lines.append("| " + " | ".join(cells) + " |")
    md = "\n".join(lines)
    Path(f"{stem}.md").write_text(md + "\n")
    print(md + "\n")


def plot_distributions(scores, taus, group, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    for ax, name in zip(axes, ("MSP", "MLS", "Mahalanobis")):
        s = scores[name]
        allv = np.concatenate([s["test"], s["unknown"]])
        bins = np.linspace(np.percentile(allv, 0.5), np.percentile(allv, 99.5), 60)
        ax.hist(s["test"], bins=bins, alpha=0.5, density=True, label="Known (CIFAR-10 test)")
        ax.hist(s["unknown"][group == GROUP_NEAR], bins=bins, alpha=0.5, density=True, label="Near unknown")
        ax.hist(s["unknown"][group == GROUP_FAR], bins=bins, alpha=0.5, density=True, label="Far unknown")
        ax.axvline(taus[name], color="k", ls="--", lw=1, label="val 95% threshold")
        ax.set_title(f"Vanilla: u_{name}")
        ax.set_xlabel("unknownness")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=str(ROOT / "datasets"))
    parser.add_argument("--ckpt_dir", type=str, default=str(ROOT / "checkpoints"))
    parser.add_argument("--cache_dir", type=str, default=str(ROOT / "cache"))
    parser.add_argument("--results_dir", type=str, default=str(ROOT / "results"))
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    results = Path(args.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    missing = [m for m in METHODS if not ckpt_path(args.ckpt_dir, m).exists()]
    if missing:
        raise SystemExit(f"Missing checkpoints for {missing}. Run train.py for each config first.")

    caches = {}
    for m in METHODS:
        ensure_known_cache(m, args.ckpt_dir, args.cache_dir, args.data_root, device, args.batch_size, args.num_workers)
        ensure_unknown_cache(m, args.ckpt_dir, args.cache_dir, args.data_root, device, args.batch_size, args.num_workers)
        caches[m] = load_cache(args.cache_dir, m, ["train", "val", "test", "unknown"])

    van = caches["vanilla"]
    mah = fit_mahalanobis(van["train"]["features"], van["train"]["labels"])
    score_fns = {"MSP": msp_score, "MLS": mls_score, "Energy": energy_score, "Mahalanobis": mah}
    van_scores, rows1 = {}, []
    for name, fn in score_fns.items():
        van_scores[name] = score_splits(fn, van)
        rows1.append(make_row({"score": name}, van_scores[name], van))
    print("Table 1: scores on frozen Vanilla model")
    write_table(rows1, ["score"], str(results / "table1_scores_vanilla"))

    rows2 = []
    for m in METHODS:
        c = caches[m]
        csa = closed_set_accuracy(c["test"]["logits"], c["test"]["labels"], NUM_KNOWN)
        sc = score_splits(mls_score, c, num_logits=NUM_KNOWN)  # MLS on the ten known-class logits only
        rows2.append(make_row({"model": m, "score": "MLS"}, sc, c, csa))
    cp = caches["proser"]
    csa_p = closed_set_accuracy(cp["test"]["logits"], cp["test"]["labels"], NUM_KNOWN)
    sc_p = score_splits(lambda lg, ft: placeholder_unknownness(lg, NUM_KNOWN), cp)  # uses all 15 logits
    rows2.append(make_row({"model": "proser", "score": "Placeholder"}, sc_p, cp, csa_p))
    print("Table 2: Vanilla vs GCSC vs PROSER")
    write_table(rows2, ["model", "score"], str(results / "table2_methods"), include_csa=True)

    unk = van["unknown"]
    tau_mls = rows1[1]["tau"]
    mls_unk = van_scores["MLS"]["unknown"]
    pred = unk["logits"][:, :NUM_KNOWN].argmax(axis=1)
    fine_names = unk["fine_names"]
    records = collect_failures(mls_unk, tau_mls, pred, unk["fine_labels"], unk["group"], fine_names,
                               CIFAR10_CLASSES, k=3)
    report = format_failures(records)
    print("Failure analysis (Vanilla MLS)\n" + report + "\n")
    (results / "failure_analysis.txt").write_text(report + "\n")
    with open(results / "failure_analysis.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    images = load_unknowns(args.data_root, allow_eval_access=True)["images"]
    save_failure_images(records, images, results / "failures")

    pc = per_class_acceptance(mls_unk, tau_mls, pred, unk["fine_labels"], unk["group"], fine_names, CIFAR10_CLASSES)
    with open(results / "per_class_acceptance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pc[0].keys()))
        w.writeheader()
        w.writerows(pc)
    print("Per-class acceptance (Vanilla MLS)")
    for r in sorted(pc, key=lambda r: -r["accepted_rate"]):
        print(f"  {r['group']:<4} {r['unknown_class']:<13} accepted={100 * r['accepted_rate']:5.1f}%  "
              f"absorbed-by={r['top_absorbing_class']}")

    taus = {r["score"]: r["tau"] for r in rows1}
    plot_distributions(van_scores, taus, unk["group"], results / "score_distributions.png")
    print(f"\nAll results written to {results}")


if __name__ == "__main__":
    main()