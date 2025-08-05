import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def main(visual_method: str = "tsne"):
    labels = (
        torch.load("test_emb_labels.pt", map_location="cpu")
        .numpy()
        .squeeze()
        .astype(np.uint8)
    )
    X = torch.load("test_embs.pt", map_location="cpu").to(torch.float32).numpy()

    print(f"X shape: {X.shape}, labels shape: {labels.shape}")

    if visual_method == "tsne":
        visual_func = TSNE()
    elif visual_method == "umap":
        visual_func = umap.UMAP()
    elif visual_method == "pca":
        visual_func = PCA(n_components=2)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_tsne = visual_func.fit_transform(X)

    palette = sns.color_palette("Set1", len(np.unique(labels)))

    plt.figure(figsize=(10, 8), dpi=120)

    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(
            X_tsne[idx, 0],
            X_tsne[idx, 1],
            label=f"Class {label}",
            s=70,
            linewidths=0.5,
            alpha=0.8,
            color=palette[0 if label == 1 else 1],
            edgecolors="none",
        )

    # Beautify
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    plt.grid(False)
    plt.legend(frameon=False, fontsize=24, loc="best")
    plt.tight_layout()
    plt.savefig(f"test_emb_{visual_method}.png", dpi=300)


if __name__ == "__main__":
    # visualization method options: tsne, pca, umap
    main(visual_method="tsne")
