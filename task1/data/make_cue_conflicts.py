import itertools
import json
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights

CONTENT_LAYER = "21"
STYLE_LAYERS = ["0", "5", "10", "19", "28"]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

EDGE_CORR_RANGE = (0.35, 0.90)
COLOR_DIST_MIN = 0.03


class VGGFeatures(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg

    def forward(self, x01):
        x = (x01 - IMAGENET_MEAN.to(x01.device)) / IMAGENET_STD.to(x01.device)
        feats = {}
        for idx, layer in enumerate(self.vgg):
            x = layer(x)
            name = str(idx)
            if name in STYLE_LAYERS or name == CONTENT_LAYER:
                feats[name] = x
            if idx == 28:
                break
        return feats


def gram_matrix(feat):
    b, c, h, w = feat.shape
    f = feat.reshape(b, c, h * w)
    g = torch.bmm(f, f.transpose(1, 2))
    return g / (c * h * w)


def style_transfer(content01, style01, vgg, num_steps=250, style_weight=1e6,
                   content_weight=1.0, device="cpu"):
    content01, style01 = content01.to(device), style01.to(device)
    with torch.no_grad():
        content_feats = vgg(content01)
        style_feats = vgg(style01)
        style_grams = {k: gram_matrix(v) for k, v in style_feats.items() if k in STYLE_LAYERS}

    image = content01.clone().requires_grad_(True)
    opt = torch.optim.Adam([image], lr=0.02)

    for _ in range(num_steps):
        opt.zero_grad()
        feats = vgg(image.clamp(0, 1))
        c_loss = F.mse_loss(feats[CONTENT_LAYER], content_feats[CONTENT_LAYER])
        s_loss = sum(F.mse_loss(gram_matrix(feats[k]), style_grams[k]) for k in STYLE_LAYERS)
        loss = content_weight * c_loss + style_weight * s_loss
        loss.backward()
        opt.step()

    return image.detach().clamp(0, 1)


def _sobel_edges(img01):
    gray = (0.299 * img01[:, 0] + 0.587 * img01[:, 1] + 0.114 * img01[:, 2]).unsqueeze(1)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32,
                        device=img01.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


def _edge_correlation(a01, b01):
    ea, eb = _sobel_edges(a01).flatten(), _sobel_edges(b01).flatten()
    ea, eb = ea - ea.mean(), eb - eb.mean()
    denom = (ea.norm() * eb.norm()).clamp_min(1e-8)
    return (ea @ eb / denom).item()


def _color_histogram_distance(a01, b01, bins=32):
    a01, b01 = a01.detach().cpu(), b01.detach().cpu()
    dists = []
    for ch in range(3):
        ha = torch.histc(a01[0, ch], bins=bins, min=0, max=1)
        hb = torch.histc(b01[0, ch], bins=bins, min=0, max=1)
        ha, hb = ha / ha.sum().clamp_min(1e-8), hb / hb.sum().clamp_min(1e-8)
        dists.append(0.5 * (ha - hb).abs().sum().item())
    return float(np.mean(dists))


def visual_rejection_rule(content01, stylized01):
    edge_corr = _edge_correlation(content01, stylized01)
    color_dist = _color_histogram_distance(content01, stylized01)
    accepted = (EDGE_CORR_RANGE[0] <= edge_corr <= EDGE_CORR_RANGE[1]) and (color_dist >= COLOR_DIST_MIN)
    return {"edge_correlation": edge_corr, "color_hist_distance": color_dist, "accepted": accepted}


def generate_cue_conflicts(dataset, indices_by_class, class_names, out_dir,
                            transform, class_pairs=None, per_pair_target=50,
                            min_valid_total=200, num_steps=250,
                            seed=6304, device="cpu"):
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)
    vgg = VGGFeatures(device=device)

    if class_pairs is None:
        all_pairs = list(itertools.combinations(range(len(class_names)), 2))
        random.shuffle(all_pairs)
        class_pairs = all_pairs[:5]

    manifest, accepted_count, rejected_count = [], 0, 0
    for (ca, cb) in class_pairs:
        idx_a = list(indices_by_class[ca])
        idx_b = list(indices_by_class[cb])
        random.shuffle(idx_a)
        random.shuffle(idx_b)
        n_pairs = min(per_pair_target, len(idx_a), len(idx_b))
        half = n_pairs // 2

        for direction, (shape_cls, style_cls, shape_idx, style_idx) in enumerate([
            (ca, cb, idx_a, idx_b), (cb, ca, idx_b, idx_a),
        ]):
            offset = direction * half
            for k in range(half):
                content_id = shape_idx[offset + k]
                style_id = style_idx[offset + k]
                content_pil, _ = dataset[content_id]
                style_pil, _ = dataset[style_id]
                content01 = transform(content_pil).unsqueeze(0)
                style01 = transform(style_pil).unsqueeze(0)

                stylized = style_transfer(content01, style01, vgg, num_steps=num_steps, device=device)
                rule = visual_rejection_rule(content01.to(device), stylized)

                record = {
                    "content_id": int(content_id), "style_id": int(style_id),
                    "shape_class": int(shape_cls), "texture_class": int(style_cls),
                    "shape_class_name": class_names[shape_cls],
                    "texture_class_name": class_names[style_cls],
                    **rule,
                }
                if rule["accepted"]:
                    fname = f"pair{ca}-{cb}_dir{direction}_c{content_id}_s{style_id}.pt"
                    torch.save(stylized.cpu(), os.path.join(out_dir, fname))
                    record["file"] = fname
                    accepted_count += 1
                else:
                    rejected_count += 1
                manifest.append(record)

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({"class_pairs": class_pairs, "accepted": accepted_count,
                    "rejected": rejected_count, "records": manifest}, f, indent=2)

    print(f"Cue conflicts: accepted {accepted_count} / rejected {rejected_count} (target >= {min_valid_total})")
    if accepted_count < min_valid_total:
        print("WARNING: fewer than the required 200 valid conflicts were produced; increase per_pair_target, add more class pairs, or raise num_steps.")
    return manifest