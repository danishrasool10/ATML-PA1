import json
import os
import sys
import numpy as np
import torch
from torchvision.datasets import STL10

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.make_subset import build_splits, load_splits, STL10Subset
from data.transforms import base_transform, to_grayscale_3ch, hue_rotate, four_direction_translations, patch_shuffle, set_all_seeds, TRANSLATIONS_PX
from data.make_cue_conflicts import generate_cue_conflicts
from models.backbones import build_backbones
from analysis.evaluate_bias import train_linear_head, evaluate_head, evaluate_clip_zero_shot, extract_features_batched, prediction_consistency, shape_bias_and_coverage
from analysis.feature_similarity import cosine_stability
from analysis.representation import plot_clean_vs_transformed

SEED = 6304
DATA_ROOT = "./data/raw"
SPLIT_DIR = "./data/splits"
CUE_CONFLICT_DIR = "./data/cue_conflicts"
RESULTS_DIR = "./results"
HEAD_MODELS = ["resnet50", "vit_b_16"]
ALL_MODELS = ["resnet50", "vit_b_16", "clip_vit_b_32"]

def stack_dataset(dataset):
    imgs, labels, ids = [], [], []
    for img, label, real_id in dataset:
        imgs.append(img)
        labels.append(label)
        ids.append(real_id)
    return torch.stack(imgs), torch.tensor(labels), torch.tensor(ids)

def apply_to_batch(images, fn):
    return torch.stack([fn(images[i]) for i in range(images.shape[0])])

def main():
    set_all_seeds(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(os.path.join(SPLIT_DIR, "splits.json")):
        build_splits(DATA_ROOT, SPLIT_DIR, SEED)
    splits = load_splits(SPLIT_DIR)
    class_names = splits["classes"]
    num_classes = len(class_names)

    stl_train = STL10(root=DATA_ROOT, split="train", download=True)
    stl_test = STL10(root=DATA_ROOT, split="test", download=True)
    tfm = base_transform()

    train_ds = STL10Subset(stl_train, splits["train_idx"], tfm)
    val_ds = STL10Subset(stl_train, splits["val_idx"], tfm)
    test_ds = STL10Subset(stl_test, splits["test_subset_idx"], tfm)

    train_imgs, train_labels, _ = stack_dataset(train_ds)
    val_imgs, val_labels, _ = stack_dataset(val_ds)
    test_imgs, test_labels, test_ids = stack_dataset(test_ds)

    backbones = build_backbones(device=device)
    results = {"clean_baseline": {}, "color_bias": {}, "translation": {}, "patch_shuffle": {}}
    heads, clean_features, clean_preds = {}, {}, {}

    for name in HEAD_MODELS:
        bb = backbones[name]
        train_f = extract_features_batched(bb, train_imgs, device=device)
        val_f = extract_features_batched(bb, val_imgs, device=device)
        test_f = extract_features_batched(bb, test_imgs, device=device)

        head = train_linear_head(train_f.to(device), train_labels.to(device),
                                 val_f.to(device), val_labels.to(device),
                                 bb.feature_dim, num_classes, device=device, seed=SEED)
        heads[name] = head
        clean_features[name] = test_f

        metrics = evaluate_head(head, test_f, test_labels, device=device)
        clean_preds[name] = metrics.pop("predictions")
        results["clean_baseline"][name] = metrics

    clip_bb = backbones["clip_vit_b_32"]
    text_feats = clip_bb.encode_class_prompts(class_names).to(device)
    clean_features["clip_vit_b_32"] = extract_features_batched(clip_bb, test_imgs, device=device)
    clip_metrics = evaluate_clip_zero_shot(clip_bb, test_imgs, text_feats, test_labels, device=device)
    clean_preds["clip_vit_b_32"] = clip_metrics.pop("predictions")
    results["clean_baseline"]["clip_vit_b_32_zero_shot"] = clip_metrics
    print("Clean baseline:", json.dumps(results["clean_baseline"], indent=2))

    gray_imgs = apply_to_batch(test_imgs, to_grayscale_3ch)
    hue_imgs = apply_to_batch(test_imgs, hue_rotate)
    for cond_name, cond_imgs in [("grayscale", gray_imgs), ("hue_rotation", hue_imgs)]:
        results["color_bias"][cond_name] = {}
        for name in HEAD_MODELS:
            bb = backbones[name]
            feats = extract_features_batched(bb, cond_imgs, device=device)
            m = evaluate_head(heads[name], feats, test_labels, device=device)
            preds = m.pop("predictions")
            m["prediction_consistency_vs_clean"] = prediction_consistency(preds, clean_preds[name])
            m["cosine_stability"] = cosine_stability(clean_features[name], feats)["mean_cosine_stability"]
            results["color_bias"][cond_name][name] = m

        feats_clip = extract_features_batched(clip_bb, cond_imgs, device=device)
        m = evaluate_clip_zero_shot(clip_bb, cond_imgs, text_feats, test_labels, device=device)
        preds = m.pop("predictions")
        m["prediction_consistency_vs_clean"] = prediction_consistency(preds, clean_preds["clip_vit_b_32"])
        m["cosine_stability"] = cosine_stability(clean_features["clip_vit_b_32"], feats_clip)["mean_cosine_stability"]
        results["color_bias"][cond_name]["clip_vit_b_32_zero_shot"] = m
    print("Color bias done.")

    indices_by_class_full = {c: [] for c in range(num_classes)}
    for idx, label in enumerate(stl_test.labels):
        indices_by_class_full[int(label)].append(idx)

    manifest = generate_cue_conflicts(stl_test, indices_by_class_full, class_names,
                                      CUE_CONFLICT_DIR, transform=tfm, seed=SEED, device=device)
    accepted = [r for r in manifest if r["accepted"]]

    conflict_imgs = shape_labels = texture_labels = None
    if accepted:
        conflict_imgs = torch.stack([
            torch.load(os.path.join(CUE_CONFLICT_DIR, r["file"])).squeeze(0) for r in accepted
        ])
        shape_labels = torch.tensor([r["shape_class"] for r in accepted])
        texture_labels = torch.tensor([r["texture_class"] for r in accepted])

        results["shape_vs_texture"] = {}
        for name in HEAD_MODELS:
            bb = backbones[name]
            feats = extract_features_batched(bb, conflict_imgs, device=device)
            logits = heads[name](feats.to(device))
            preds = logits.argmax(-1).cpu().numpy()
            shape_flags = preds == shape_labels.numpy()
            texture_flags = preds == texture_labels.numpy()
            results["shape_vs_texture"][name] = shape_bias_and_coverage(shape_flags, texture_flags)

        logits = clip_bb.zero_shot_logits(conflict_imgs.to(device), text_feats).cpu()
        preds = logits.argmax(-1).numpy()
        shape_flags = preds == shape_labels.numpy()
        texture_flags = preds == texture_labels.numpy()
        results["shape_vs_texture"]["clip_vit_b_32_zero_shot"] = shape_bias_and_coverage(shape_flags, texture_flags)
    print("Shape-vs-texture done. Accepted:", len(accepted))

    for px in TRANSLATIONS_PX:
        results["translation"][px] = {n: {"acc": [], "cons": [], "stab": []} for n in
                                      HEAD_MODELS + ["clip_vit_b_32_zero_shot"]}
        for direction in ["up", "down", "left", "right"]:
            trans_imgs = torch.stack([
                four_direction_translations(test_imgs[i], px)[direction] for i in range(test_imgs.shape[0])
            ])
            for name in HEAD_MODELS:
                bb = backbones[name]
                feats = extract_features_batched(bb, trans_imgs, device=device)
                m = evaluate_head(heads[name], feats, test_labels, device=device)
                preds = m.pop("predictions")
                results["translation"][px][name]["acc"].append(m["top1_accuracy"])
                results["translation"][px][name]["cons"].append(prediction_consistency(preds, clean_preds[name]))
                results["translation"][px][name]["stab"].append(
                    cosine_stability(clean_features[name], feats)["mean_cosine_stability"])

            m = evaluate_clip_zero_shot(clip_bb, trans_imgs, text_feats, test_labels, device=device)
            preds = m.pop("predictions")
            feats_c = extract_features_batched(clip_bb, trans_imgs, device=device)
            results["translation"][px]["clip_vit_b_32_zero_shot"]["acc"].append(m["top1_accuracy"])
            results["translation"][px]["clip_vit_b_32_zero_shot"]["cons"].append(
                prediction_consistency(preds, clean_preds["clip_vit_b_32"]))
            results["translation"][px]["clip_vit_b_32_zero_shot"]["stab"].append(
                cosine_stability(clean_features["clip_vit_b_32"], feats_c)["mean_cosine_stability"])

        for name, d in results["translation"][px].items():
            results["translation"][px][name] = {
                "mean_accuracy": float(np.mean(d["acc"])),
                "mean_consistency": float(np.mean(d["cons"])),
                "mean_cosine_stability": float(np.mean(d["stab"])),
            }
    print("Translation done.")

    shuffled_imgs = torch.stack([
        patch_shuffle(test_imgs[i], int(test_ids[i].item()), seed=SEED) for i in range(test_imgs.shape[0])
    ])
    for name in HEAD_MODELS:
        bb = backbones[name]
        feats = extract_features_batched(bb, shuffled_imgs, device=device)
        m = evaluate_head(heads[name], feats, test_labels, device=device)
        preds = m.pop("predictions")
        m["prediction_consistency_vs_clean"] = prediction_consistency(preds, clean_preds[name])
        m["cosine_stability"] = cosine_stability(clean_features[name], feats)["mean_cosine_stability"]
        results["patch_shuffle"][name] = m

    m = evaluate_clip_zero_shot(clip_bb, shuffled_imgs, text_feats, test_labels, device=device)
    preds = m.pop("predictions")
    feats_c = extract_features_batched(clip_bb, shuffled_imgs, device=device)
    m["prediction_consistency_vs_clean"] = prediction_consistency(preds, clean_preds["clip_vit_b_32"])
    m["cosine_stability"] = cosine_stability(clean_features["clip_vit_b_32"], feats_c)["mean_cosine_stability"]
    results["patch_shuffle"]["clip_vit_b_32_zero_shot"] = m
    print("Patch shuffle done.")

    results["representation_stability"] = {}
    translated_32_right = torch.stack([
        four_direction_translations(test_imgs[i], 32)["right"] for i in range(test_imgs.shape[0])
    ])
    repr_interventions = {
        "grayscale": gray_imgs,
        "patch_shuffle": shuffled_imgs,
        "translation_32px": translated_32_right,
    }

    clean_content_imgs = None
    if accepted:
        clean_content_imgs = torch.stack([tfm(stl_test[r["content_id"]][0]) for r in accepted])

    for name in ALL_MODELS:
        bb = backbones[name]
        results["representation_stability"][name] = {}
        for cond_name, cond_imgs in repr_interventions.items():
            feats_t = extract_features_batched(bb, cond_imgs, device=device)
            stab = cosine_stability(clean_features[name], feats_t)["mean_cosine_stability"]
            results["representation_stability"][name][cond_name] = stab
            plot_clean_vs_transformed(
                clean_features[name].numpy(), feats_t.numpy(), test_labels.numpy(), class_names,
                f"{name}: clean vs {cond_name}",
                os.path.join(RESULTS_DIR, "figures", f"{name}_{cond_name}_proj.png"), seed=SEED)

        if accepted:
            clean_content_feats = extract_features_batched(bb, clean_content_imgs, device=device)
            conflict_feats = extract_features_batched(bb, conflict_imgs, device=device)
            stab = cosine_stability(clean_content_feats, conflict_feats)["mean_cosine_stability"]
            results["representation_stability"][name]["cue_conflict"] = stab
            plot_clean_vs_transformed(
                clean_content_feats.numpy(), conflict_feats.numpy(), shape_labels.numpy(), class_names,
                f"{name}: clean content vs cue-conflict",
                os.path.join(RESULTS_DIR, "figures", f"{name}_cue_conflict_proj.png"), seed=SEED)
    print("Representation analysis done.")

    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Done. Results saved to", RESULTS_DIR)

if __name__ == "__main__":
    main()