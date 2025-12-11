import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



NUM_CLASSES = 10


CLASS_NAMES = [
    "air_conditioner",   # 0
    "car_horn",          # 1
    "children_playing",  # 2
    "dog_bark",          # 3
    "drilling",          # 4
    "engine_idling",     # 5
    "gun_shot",          # 6
    "jackhammer",        # 7
    "siren",             # 8
    "street_music",      # 9
]


TRAIN_FOLDS = [1, 2, 3, 4, 5, 6, 7, 8]
VAL_FOLDS = [9]
TEST_FOLDS = [10]

# functions


def set_seed(seed: int = 42):
    """固定随机种子，保证可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Dataset

class EmbeddingDataset(Dataset):
    """从 YAMNet 提取好的 embedding 中读取数据。"""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        """
        embeddings: [N, D] float32
        labels:     [N] int64
        """
        self.embeddings = embeddings.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.embeddings[idx]   # [D]
        y = self.labels[idx]
        return torch.from_numpy(x), torch.tensor(y)


# Model definition

#Simple two-layer MLP classifier operating on 1024-dim YAMNet embeddings.
class YAMNetMLP(nn.Module):


    def __init__(self, in_dim: int = 1024, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# Fold-based split

def split_by_folds(
    embeddings: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    train_folds,
    val_folds,
    test_folds,
):
    #Split embeddings/labels by UrbanSound8K folds into train/val/test sets.
    train_mask = np.isin(folds, train_folds)
    val_mask = np.isin(folds, val_folds)
    test_mask = np.isin(folds, test_folds)

    X_train, y_train = embeddings[train_mask], labels[train_mask]
    X_val, y_val = embeddings[val_mask], labels[val_mask]
    X_test, y_test = embeddings[test_mask], labels[test_mask]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# Training & evaluation

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

#Evaluate the model on the given DataLoader and return metrics.
def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += x.size(0)

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    avg_loss = total_loss / total
    acc = correct / total

    return avg_loss, acc, all_labels, all_preds


def plot_training_curves(history, save_path="plots/yamnet_training_curves.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 4))

    # 左边：Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], marker="s", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve(YAMNet)")
    plt.legend()
    plt.grid(alpha=0.3)

    # 右边：Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], marker="o", label="Train Acc")
    plt.plot(epochs, history["val_acc"], marker="s", label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve(YAMNet)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Training curves saved to {save_path}")

#Plot two confusion matrices for YAMNet + MLP
def plot_confusion_matrices_yamnet(y_true, y_pred, class_names, save_dir="plots"):

    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    # raw counts
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("UrbanSound8K - Confusion Matrix (YAMNet + MLP)")
    plt.tight_layout()
    cm_path = os.path.join(save_dir, "yamnet_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    print(f"Confusion matrix (YAMNet counts) saved to {cm_path}")
    plt.close()

    # row-normalized
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        vmin=0.0,
        vmax=1.0,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("UrbanSound8K - Normalized Confusion Matrix (YAMNet + MLP)")
    plt.tight_layout()
    cm_norm_path = os.path.join(save_dir, "yamnet_confusion_matrix_normalized.png")
    plt.savefig(cm_norm_path, dpi=300)
    print(f"Confusion matrix (YAMNet normalized) saved to {cm_norm_path}")
    plt.close()



def main(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load NPZ: embeddings / labels / folds
    data = np.load(args.npz_path)
    embeddings = data["embeddings"]  # [N, D]
    labels = data["labels"]          # [N]
    folds = data["folds"]            # [N]

    print("Loaded embeddings from:", args.npz_path)
    print("Embeddings shape:", embeddings.shape)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_by_folds(
        embeddings, labels, folds,
        TRAIN_FOLDS, VAL_FOLDS, TEST_FOLDS
    )

    print("Train size:", X_train.shape[0])
    print("Val size:", X_val.shape[0])
    print("Test size:", X_test.shape[0])

    # ---- DataLoader ----
    train_dataset = EmbeddingDataset(X_train, y_train)
    val_dataset = EmbeddingDataset(X_val, y_val)
    test_dataset = EmbeddingDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Model / loss / optimizer
    in_dim = embeddings.shape[1]
    model = YAMNetMLP(in_dim=in_dim, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    best_val_acc = 0.0
    best_state = None

    # History for plotting curves
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = eval_model(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"[Epoch {epoch:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Track best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    print("Best val acc:", best_val_acc)

    # Plot training curves
    plot_training_curves(history, save_path=args.save_curves)

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test
    test_loss, test_acc, y_true, y_pred = eval_model(
        model, test_loader, criterion, device
    )
    print("\n===== Test result (YAMNet + MLP) =====")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    # classification_report
    print("\nTest Report:")
    report_str = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        digits=3,
    )
    print(report_str)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    print("Confusion matrix (rows = true, cols = pred):")
    print(cm)

    plot_confusion_matrices_yamnet(y_true, y_pred, CLASS_NAMES, save_dir="plots")

    if args.save_report:
        with open(args.save_report, "w", encoding="utf-8") as f:
            f.write("===== Test result (YAMNet + MLP) =====\n")
            f.write(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}\n\n")
            f.write("Test Report:\n")
            f.write(report_str)
            f.write("\nConfusion matrix (rows = true, cols = pred):\n")
            np.savetxt(f, cm, fmt="%d")
        print(f"\nReport saved to: {args.save_report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a small MLP classifier on YAMNet embeddings for UrbanSound8K."
    )
    parser.add_argument(
        "--npz_path",
        type=str,
        default="yamnet_urbansound8k_embeddings.npz",
        help="Path to NPZ file with embeddings/labels/folds.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--save_report",
        type=str,
        default="yamnet_test_report.txt",
        help="Path to save text report (set empty string to disable).",
    )
    parser.add_argument(
        "--save_curves",
        type=str,
        default="plots/yamnet_training_curves.png",
        help="Path to save training curves figure.",
    )

    args = parser.parse_args()
    if args.save_report == "":
        args.save_report = None

    main(args)
