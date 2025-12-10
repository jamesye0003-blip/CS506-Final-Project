import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix


# =====================
# 配置 & 常量
# =====================

NUM_CLASSES = 10

# UrbanSound8K 官方 classID → 类别名称
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

# 默认 fold 划分：1–8 训练，9 验证，10 测试
TRAIN_FOLDS = [1, 2, 3, 4, 5, 6, 7, 8]
VAL_FOLDS = [9]
TEST_FOLDS = [10]


# =====================
# 工具函数
# =====================

def set_seed(seed: int = 42):
    """固定随机种子，保证可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =====================
# Dataset 定义
# =====================

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


# =====================
# 模型定义
# =====================

class YAMNetMLP(nn.Module):
    """简单的两层 MLP，用 YAMNet 1024 维 embedding 做分类。"""

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


# =====================
# 数据划分
# =====================

def split_by_folds(
    embeddings: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    train_folds,
    val_folds,
    test_folds,
):
    """按 UrbanSound8K 的 fold 划分 train/val/test。"""
    train_mask = np.isin(folds, train_folds)
    val_mask = np.isin(folds, val_folds)
    test_mask = np.isin(folds, test_folds)

    X_train, y_train = embeddings[train_mask], labels[train_mask]
    X_val, y_val = embeddings[val_mask], labels[val_mask]
    X_test, y_test = embeddings[test_mask], labels[test_mask]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# =====================
# 训练 & 验证
# =====================

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


# =====================
# 主程序
# =====================

def main(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---- 读取 NPZ：embeddings / labels / folds ----
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

    # ---- 模型 / 损失 / 优化器 ----
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

    # ---- 训练循环 ----
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = eval_model(
            model, val_loader, criterion, device
        )

        print(
            f"[Epoch {epoch:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # 记录最好的验证集模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    print("Best val acc:", best_val_acc)

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Test 评估 ----
    test_loss, test_acc, y_true, y_pred = eval_model(
        model, test_loader, criterion, device
    )
    print("\n===== Test result (YAMNet + MLP) =====")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    # classification_report（就是你截图里的那个表）
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

    # 可选：把报告保存成文本文件，方便写 README
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

    args = parser.parse_args()
    if args.save_report == "":
        args.save_report = None

    main(args)