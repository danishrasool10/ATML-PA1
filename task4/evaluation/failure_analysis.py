"""Incorrectly accepted unknowns under the Vanilla-MLS threshold."""
from pathlib import Path

import numpy as np
from PIL import Image

from data.cifar100_unknowns import GROUP_FAR, GROUP_NEAR


def collect_failures(scores, tau, pred_labels, fine_labels, group, fine_names, cifar10_names, k=3):
    """Return k accepted unknowns per group, preferring distinct unknown classes."""
    records = []
    for gid, gname in ((GROUP_NEAR, "near"), (GROUP_FAR, "far")):
        cand = np.where((group == gid) & (scores <= tau))[0]
        cand = cand[np.argsort(scores[cand], kind="stable")]
        chosen, seen = [], set()
        for i in cand:
            if fine_labels[i] not in seen:
                chosen.append(i)
                seen.add(fine_labels[i])
            if len(chosen) == k:
                break
        for i in cand:
            if len(chosen) == k:
                break
            if i not in chosen:
                chosen.append(i)
        for i in chosen:
            records.append({"group": gname, "unknown_index": int(i),
                            "unknown_class": str(fine_names[fine_labels[i]]),
                            "predicted_class": cifar10_names[int(pred_labels[i])],
                            "score": float(scores[i]), "threshold": float(tau)})
    return records


def per_class_acceptance(scores, tau, pred_labels, fine_labels, group, fine_names, cifar10_names):
    """Return per-class acceptance rates and top absorbing CIFAR-10 class."""
    rows = []
    for gid, gname in ((GROUP_NEAR, "near"), (GROUP_FAR, "far")):
        for c in np.unique(fine_labels[group == gid]):
            m = (fine_labels == c) & (group == gid)
            acc = m & (scores <= tau)
            top = ""
            if acc.any():
                counts = np.bincount(pred_labels[acc], minlength=len(cifar10_names))
                top = cifar10_names[int(counts.argmax())]
            rows.append({"group": gname, "unknown_class": str(fine_names[c]), "n": int(m.sum()),
                         "accepted_rate": float(acc.sum() / m.sum()), "top_absorbing_class": top})
    return rows


def format_failures(records):
    lines = [f"{'group':<5} {'unknown class':<14} {'predicted CIFAR-10':<18} {'score(-max logit)':>18} {'threshold':>10}"]
    for r in records:
        lines.append(f"{r['group']:<5} {r['unknown_class']:<14} {r['predicted_class']:<18} "
                     f"{r['score']:>18.4f} {r['threshold']:>10.4f}")
    return "\n".join(lines)


def save_failure_images(records, images, out_dir, scale=4):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for j, r in enumerate(records):
        img = Image.fromarray(images[r["unknown_index"]]).resize((32 * scale, 32 * scale), Image.NEAREST)
        img.save(out_dir / f"{r['group']}_{j:02d}_{r['unknown_class']}_pred-{r['predicted_class']}.png")