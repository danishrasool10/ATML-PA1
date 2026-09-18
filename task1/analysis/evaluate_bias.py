import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from models.backbones import LinearHead

SEED = 6304
MAX_EPOCHS = 50
PATIENCE = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def compute_metrics(logits, labels):
    probs = torch.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1)
    conf = probs.max(dim=-1).values
    acc = (preds == labels).float().mean().item()
    f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average="macro")
    return {
        "top1_accuracy": acc,
        "macro_f1": float(f1),
        "mean_max_confidence": conf.mean().item(),
        "predictions": preds.cpu().numpy(),
    }


def train_linear_head(train_feats, train_labels, val_feats, val_labels,
                      feature_dim, num_classes, device="cpu",
                      seed=SEED, batch_size=BATCH_SIZE):
    set_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    head = LinearHead(feature_dim, num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    n = train_feats.shape[0]
    best_val_acc, best_state, epochs_no_improve = -1.0, None, 0

    for _ in range(MAX_EPOCHS):
        head.train()
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            logits = head(train_feats[idx].to(device))
            loss = loss_fn(logits, train_labels[idx].to(device))
            loss.backward()
            opt.step()

        head.eval()
        with torch.no_grad():
            val_logits = head(val_feats.to(device))
            val_acc = (val_logits.argmax(-1) == val_labels.to(device)).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc, epochs_no_improve = val_acc, 0
            best_state = copy.deepcopy(head.state_dict())
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= PATIENCE:
            break

    head.load_state_dict(best_state)
    head.eval()
    return head


@torch.no_grad()
def extract_features_batched(backbone, images_01, batch_size=64, device="cpu"):
    feats = []
    for i in range(0, images_01.shape[0], batch_size):
        batch = images_01[i:i + batch_size].to(device)
        feats.append(backbone.extract_features(batch).cpu())
    return torch.cat(feats, dim=0)


@torch.no_grad()
def evaluate_head(head, feats, labels, device="cpu"):
    head.eval()
    logits = head(feats.to(device))
    return compute_metrics(logits.cpu(), labels)


@torch.no_grad()
def evaluate_clip_zero_shot(clip_backbone, images_01, text_features,
                            labels, batch_size=64, device="cpu"):
    all_logits = []
    for i in range(0, images_01.shape[0], batch_size):
        batch = images_01[i:i + batch_size].to(device)
        all_logits.append(clip_backbone.zero_shot_logits(batch, text_features).cpu())
    logits = torch.cat(all_logits, dim=0)
    return compute_metrics(logits, labels)


def prediction_consistency(preds_a, preds_b):
    return float(np.mean(preds_a == preds_b))


def shape_bias_and_coverage(shape_pred_flags, texture_pred_flags):
    n_shape = int(shape_pred_flags.sum())
    n_texture = int(texture_pred_flags.sum())
    n_total = len(shape_pred_flags)
    denom = n_shape + n_texture
    shape_bias = 100.0 * n_shape / denom if denom > 0 else float("nan")
    coverage = 100.0 * denom / n_total if n_total > 0 else float("nan")
    return {
        "n_shape": n_shape,
        "n_texture": n_texture,
        "n_total": n_total,
        "shape_bias_pct": shape_bias,
        "coverage_pct": coverage,
    }