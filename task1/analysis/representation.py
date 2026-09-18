import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

try:
    import umap
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False


def fit_projection(features, method="tsne", seed=6304, **kwargs):
    if method == "umap":
        if not _HAS_UMAP:
            raise ImportError("pip install umap-learn")
        reducer = umap.UMAP(
            n_components=2,
            random_state=seed,
            n_neighbors=kwargs.get("n_neighbors", 15),
            min_dist=kwargs.get("min_dist", 0.1),
        )
    elif method == "tsne":
        reducer = TSNE(
            n_components=2,
            random_state=seed,
            perplexity=kwargs.get("perplexity", 30),
            init="pca",
            learning_rate="auto",
        )
    else:
        raise ValueError(f"unknown method: {method}")

    return reducer.fit_transform(features)


def plot_clean_vs_transformed(feat_clean, feat_transformed, labels, class_names, 
                              title, out_path, method="tsne", seed=6304):
    combined = np.concatenate([feat_clean, feat_transformed], axis=0)
    proj = fit_projection(combined, method=method, seed=seed)
    n = feat_clean.shape[0]
    proj_clean, proj_trans = proj[:n], proj[n:]

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")

    for c in np.unique(labels):
        mask = labels == c
        color = cmap(int(c) % 10)
        ax.scatter(proj_clean[mask, 0], proj_clean[mask, 1], marker="o", s=18,
                   color=color, alpha=0.75)
        ax.scatter(proj_trans[mask, 0], proj_trans[mask, 1], marker="x", s=22,
                   color=color, alpha=0.75)

    class_handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=cmap(int(c) % 10),
                   label=str(class_names[c]), markersize=8)
        for c in np.unique(labels)
    ]
    condition_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", label="clean", markersize=8),
        plt.Line2D([0], [0], marker="x", color="gray", label="transformed", markersize=8),
    ]

    leg1 = ax.legend(handles=condition_handles, loc="upper right", title="condition")
    ax.add_artist(leg1)
    ax.legend(handles=class_handles, loc="upper left", title="class", fontsize=7)

    ax.set_title(f"{title} ({method})")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path