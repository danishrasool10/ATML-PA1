"""PROSER: classifier placeholders, manifold mixup, and dummy classifiers."""
import json
import time

import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from methods.manifold_mixup import manifold_mixup
from methods.vanilla import (build_loaders, evaluate_accuracy, get_device, resolve_path,
                             save_checkpoint, set_seed)
from models.resnet_cifar import ResNet18Cifar


def build_proser_model(cfg, device):
    """Initialize from Vanilla checkpoint and append random dummy classifier rows."""
    K = int(cfg["model"]["num_known"])
    M = int(cfg["model"]["num_dummy"])
    ckpt = torch.load(resolve_path(cfg["init_checkpoint"]), map_location="cpu")
    van_sd = ckpt["model"]
    assert van_sd["fc.weight"].shape[0] == K, "Vanilla checkpoint must be a 10-class model"
    model = ResNet18Cifar(num_classes=K + M)
    sd = dict(van_sd)
    fc_w, fc_b = model.fc.weight.data.clone(), model.fc.bias.data.clone()
    fc_w[:K], fc_b[:K] = van_sd["fc.weight"], van_sd["fc.bias"]
    sd["fc.weight"], sd["fc.bias"] = fc_w, fc_b
    model.load_state_dict(sd)
    return model.to(device)


def placeholder_logits(logits, num_known, exclude_labels=None):
    """Return known logits (optionally true-class masked) plus max dummy logit."""
    logits = logits.float()
    known = logits[:, :num_known]
    dummy = logits[:, num_known:].max(dim=1, keepdim=True).values
    if exclude_labels is not None:
        mask = F.one_hot(exclude_labels, num_known).bool()
        known = known.masked_fill(mask, -1e9)
    return torch.cat([known, dummy], dim=1)


def classifier_placeholder_loss(logits, labels, num_known, beta):
    """CE on known logits plus beta times CE toward dummy placeholder."""
    ce = F.cross_entropy(logits.float(), labels)
    target = torch.full_like(labels, num_known)
    cp = F.cross_entropy(placeholder_logits(logits, num_known, exclude_labels=labels), target)
    return ce + beta * cp, ce, cp


def data_placeholder_loss(logits_mix, valid, num_known):
    """CE toward dummy classifiers for manifold-mixup features."""
    target = torch.full((logits_mix.size(0),), num_known, dtype=torch.long, device=logits_mix.device)
    per = F.cross_entropy(placeholder_logits(logits_mix, num_known), target, reduction="none")
    return (per * valid.float()).sum() / valid.float().sum().clamp(min=1.0)


def placeholder_unknownness(logits, num_known=10):
    """PROSER unknownness = max dummy logit - max known logit."""
    return logits[:, num_known:].max(axis=1) - logits[:, :num_known].max(axis=1)


def run(cfg):
    set_seed(int(cfg["seed"]))
    device = get_device()
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    K = int(cfg["model"]["num_known"])
    beta = float(cfg["proser"]["beta"])
    gamma = float(cfg["proser"]["gamma"])
    alpha = float(cfg["proser"]["mixup_alpha"])
    train_loader, val_loader = build_loaders(cfg, randaugment=False)

    model = build_proser_model(cfg, device)
    o = cfg["optim"]
    epochs = int(o["epochs"])
    optimizer = SGD(model.parameters(), lr=float(o["lr"]), momentum=float(o["momentum"]),
                    weight_decay=float(o["weight_decay"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = resolve_path(cfg["output_dir"]) / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc, history = -1.0, []

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        sums = {"loss": 0.0, "ce": 0.0, "cp": 0.0, "dp": 0.0}
        correct, n, nb = 0, 0, 0
        for x, y in train_loader:
            B = x.size(0)
            if B < 4:
                continue
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            half = B // 2
            x1, y1, x2, y2 = x[:half], y[:half], x[half:], y[half:]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits1 = model(x1)
                h = model.forward_pre(x2)
                h_mix, valid = manifold_mixup(h, y2, alpha)
                logits_mix = model.forward_from_pre(h_mix)
            l1, ce, cp = classifier_placeholder_loss(logits1, y1, K, beta)
            dp = data_placeholder_loss(logits_mix, valid, K)
            loss = l1 + gamma * dp
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            sums["loss"] += loss.item(); sums["ce"] += ce.item()
            sums["cp"] += cp.item(); sums["dp"] += dp.item()
            correct += (logits1[:, :K].argmax(1) == y1).sum().item()
            n += y1.size(0)
            nb += 1
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        val_acc = evaluate_accuracy(model, val_loader, device, num_known=K)
        rec = {"epoch": epoch, "lr": lr, "val_acc": val_acc, "train_acc_half1": correct / n,
               **{k: v / nb for k, v in sums.items()}}
        history.append(rec)
        improved = val_acc > best_acc
        if improved:
            best_acc = val_acc
            save_checkpoint(out_dir / "best.pt", model, epoch, val_acc, cfg)
        print(f"[proser] epoch {epoch:3d}/{epochs} lr={lr:.6f} loss={rec['loss']:.4f} ce={rec['ce']:.4f} "
              f"cls_ph={rec['cp']:.4f} data_ph={rec['dp']:.4f} val_acc={val_acc:.4f}"
              f"{' *' if improved else ''} ({time.time() - t0:.0f}s)", flush=True)
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=1)

    print(f"[proser] best CIFAR-10 val acc {best_acc:.4f} -> {out_dir / 'best.pt'}")
    return best_acc