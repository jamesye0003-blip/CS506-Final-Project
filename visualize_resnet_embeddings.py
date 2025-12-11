import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


os.environ["LOKY_MAX_CPU_COUNT"] = "8"


try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Configuration
NPZ_PATH = "resnet3ch_urbansound8k_embeddings.npz"
SAVE_DIR = "visualizations"
SUBSAMPLE = 2000  # Subsample for t-SNE to improve speed

CLASS_NAMES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_interactive_tsne(X_2d, y):
    if not HAS_PLOTLY:
        print("Plotly not installed, skip interactive HTML.")
        return
    labels = [CLASS_NAMES[int(i)] for i in y]
    fig = px.scatter(
        x=X_2d[:, 0],
        y=X_2d[:, 1],
        color=labels,
        hover_name=labels,
        title="t-SNE of ResNet-3ch Embeddings (Interactive)",
    )
    out_html = os.path.join(SAVE_DIR, "tsne_resnet3ch_embeddings.html")
    fig.write_html(out_html)
    print(f"[Saved] {out_html}")


def main():
    ensure_dir(SAVE_DIR)

    data = np.load(NPZ_PATH)
    embeddings = data["embeddings"]   # (N, 256)
    labels = data["labels"]           # (N,)

    print("Embeddings shape:", embeddings.shape)
    print("Labels shape:", labels.shape)

    N = embeddings.shape[0]
    if SUBSAMPLE < N:
        idx = np.random.choice(N, SUBSAMPLE, replace=False)
        X = embeddings[idx]
        y = labels[idx]
    else:
        X = embeddings
        y = labels

    print("Running PCA…")
    # 先降到最多 50 维，再 t-SNE
    pca = PCA(n_components=min(50, X.shape[1]))
    X_pca = pca.fit_transform(X)

    print("Running t-SNE…")
    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=30,
    )
    X_2d = tsne.fit_transform(X_pca)

    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")
    for cid in range(10):
        mask = (y == cid)
        plt.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            s=10,
            alpha=0.6,
            color=cmap(cid),
            label=CLASS_NAMES[cid],
        )
    plt.legend()
    plt.title("t-SNE of ResNet-3ch Embeddings")
    out_png = os.path.join(SAVE_DIR, "tsne_resnet3ch_embeddings.png")
    plt.savefig(out_png, dpi=300)
    plt.show()
    print(f"[Saved] {out_png}")

    # Save interactive HTML version
    save_interactive_tsne(X_2d, y)


if __name__ == "__main__":
    main()
