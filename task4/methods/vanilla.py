"""Vanilla closed-set training and shared training utilities."""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data.cifar10 import get_cifar10_datasets
from models.resnet_cifar import ResNet18Cifar

ROOT = Path(__file__).resolve().parent.parent


def resolve_path(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loaders(cfg, randaugment):
    aug = cfg.get("augmentation", {})
    ds = get_cifar10_datasets(
        data_root=resolve_path(cfg["data_root"]),
        randaugment=randaugment,
        ra_num_ops=int(aug.get("ra_num_ops", 2)),
        ra_magnitude=int(aug.get("ra_magnitude", 9)),
        seed=int(cfg["seed"]),
    )
    nw = int(cfg.get("num_workers", 4))
    pin = torch.cuda.is_available()
    g = torch.Generator()
    g.manual_seed(int(cfg["seed"]))
    train_loader = DataLoader(ds["train"], batch_size=int(cfg["optim"]["batch_size"]), shuffle=True,
                              num_workers=nw, pin_memory=pin, generator=g, persistent_workers=nw > 0)
    val_loader = DataLoader(ds["val"], batch_size=512, shuffle=False, num_workers=nw, pin_memory=pin)
    return train_loader, val_loader


@torch.no_grad()
def evaluate_accuracy(model, loader, device, num_known=10):
    """CIFAR-10 accuracy using only the first `num_known` logits."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)[:, :num_known]
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def save_checkpoint(path, model, epoch, val_acc, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "epoch": epoch,
        "val_acc": val_acc,
        "num_classes": model.num_classes,
        "method": cfg["method"],
        "config_json": json.dumps(cfg),
    }, path)


def run_training(cfg, randaugment=False):
    set_seed(int(cfg["seed"]))
    device = get_device()
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    train_loader, val_loader = build_loaders(cfg, randaugment)

    model = ResNet18Cifar(num_classes=int(cfg["model"]["num_classes"])).to(device)
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
        loss_sum, correct, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            loss = F.cross_entropy(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        val_acc = evaluate_accuracy(model, val_loader, device, num_known=int(cfg["model"]["num_classes"]))
        history.append({"epoch": epoch, "lr": lr, "train_loss": loss_sum / n,
                        "train_acc": correct / n, "val_acc": val_acc})
        improved = val_acc > best_acc
        if improved:
            best_acc = val_acc
            save_checkpoint(out_dir / "best.pt", model, epoch, val_acc, cfg)
        print(f"[{cfg['name']}] epoch {epoch:3d}/{epochs} lr={lr:.5f} loss={loss_sum / n:.4f} "
              f"train_acc={correct / n:.4f} val_acc={val_acc:.4f}{' *' if improved else ''} "
              f"({time.time() - t0:.0f}s)", flush=True)
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=1)

    print(f"[{cfg['name']}] best CIFAR-10 val acc {best_acc:.4f} -> {out_dir / 'best.pt'}")
    return best_acc


def run(cfg):
    return run_training(cfg, randaugment=bool(cfg.get("augmentation", {}).get("randaugment", False)))