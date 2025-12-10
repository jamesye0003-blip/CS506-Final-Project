import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import classification_report, confusion_matrix


# ===================== 配置 =====================

DATA_ROOT = Path("data")
AUDIO_DIR = DATA_ROOT / "build" / "audio"
META_FILE = DATA_ROOT / "metadata" / "UrbanSound8K.csv"

# UrbanSound8K 官方类顺序
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

CFG = {
    "epochs": 80,
    "batch_size": 32,
    "lr": 1e-4,
    "weight_decay": 3e-4,
    "sample_rate": None,      # None = 保留原始 sr
    "n_mels": 64,
    "n_fft": 1024,
    "hop_length": 512,
    "fixed_seconds": 4.0,     # 统一到 4 秒
    "fmax": 8000,
    "num_workers": 2,         # 有 main guard，可以用多进程
    "seed": 42,
    "early_stop_patience": 20,
    "T_fixed": None,          # 运行时自动计算
}

TRAIN_FOLDS = [1, 2, 3, 4, 5, 6, 7, 8]
VAL_FOLDS = [9]
TEST_FOLDS = [10]


# ===================== 工具函数 =====================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def frames_for_seconds(seconds: float, hop_length: int, sr: int) -> int:
    return int(np.ceil(seconds * sr / hop_length))


# ===================== Dataset：3 通道 Mel+Δ+ΔΔ =====================

class UrbanSoundMel3Ch(Dataset):
    def __init__(self, df, audio_dir, folds, cfg, class_to_idx, is_train: bool):
        self.df = df[df["fold"].isin(folds)].reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.cfg = cfg
        self.class_to_idx = class_to_idx
        self.n_mels = cfg["n_mels"]
        self.is_train = is_train

        # 共享 T_fixed：第一次根据真实 sr 计算，写回 cfg
        if cfg["T_fixed"] is None:
            ex_row = self.df.iloc[0]
            ex_path = self.audio_dir / f"fold{ex_row.fold}" / ex_row.slice_file_name
            y, sr = sf.read(ex_path)
            y = np.asarray(y, dtype=np.float32)
            if y.ndim == 2:
                y = y.mean(axis=1)
            if cfg["sample_rate"] is not None and sr != cfg["sample_rate"]:
                y = librosa.resample(y, orig_sr=sr, target_sr=cfg["sample_rate"])
                sr = cfg["sample_rate"]
            cfg["T_fixed"] = frames_for_seconds(
                cfg["fixed_seconds"], cfg["hop_length"], sr
            )
        self.T_fixed = cfg["T_fixed"]
        print(f"[UrbanSoundMel3Ch][{'train' if is_train else 'val/test'}] T_fixed={self.T_fixed}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = self.audio_dir / f"fold{row.fold}" / row.slice_file_name

        # 读音频
        y, sr = sf.read(wav_path)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 2:
            y = y.mean(axis=1)

        # 可选重采样
        if self.cfg["sample_rate"] is not None and sr != self.cfg["sample_rate"]:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.cfg["sample_rate"])
            sr = self.cfg["sample_rate"]

        # ===== 1) 计算 log-Mel =====
        S = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=self.cfg["n_fft"],
            hop_length=self.cfg["hop_length"],
            n_mels=self.cfg["n_mels"],
            fmax=self.cfg["fmax"],
        )
        logmel = librosa.power_to_db(S, ref=np.max)  # (n_mels, T_raw)

        # ===== 2) 确保时间帧数足够做 delta（至少 9 帧）=====
        min_frames_for_delta = 9
        T_raw = logmel.shape[1]
        if T_raw < min_frames_for_delta:
            pad = min_frames_for_delta - T_raw
            # 在时间轴右侧用最后一帧做边缘填充，避免出现全 0
            logmel = np.pad(logmel, ((0, 0), (0, pad)), mode="edge")

        # 现在 logmel.shape[1] >= 9，可以安全地用默认 width=9
        delta1 = librosa.feature.delta(logmel)           # (n_mels, T_pad)
        delta2 = librosa.feature.delta(logmel, order=2)  # (n_mels, T_pad)

        # ===== 3) 组装 3 通道特征 =====
        feat = np.stack([logmel, delta1, delta2], axis=0)  # (3, n_mels, T_pad)
        _, _, T = feat.shape

        # 裁剪 / 填充到 T_fixed
        if T < self.T_fixed:
            pad_T = self.T_fixed - T
            feat = np.pad(feat, ((0, 0), (0, 0), (0, pad_T)), mode="constant")
        elif T > self.T_fixed:
            if self.is_train:
                start = np.random.randint(0, T - self.T_fixed + 1)
            else:
                start = 0
            feat = feat[:, :, start:start + self.T_fixed]

        x = feat.astype(np.float32)
        y_label = self.class_to_idx[row["class"]]

        return torch.from_numpy(x), torch.tensor(y_label, dtype=torch.long), str(wav_path)


# ===================== ResNet 模型 =====================

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + self.downsample(identity)
        out = self.relu(out)
        return out


class SmallResNet(nn.Module):
    def __init__(self, n_classes, in_channels=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = BasicBlock(32, 64, stride=2)
        self.layer2 = BasicBlock(64, 128, stride=2)
        self.layer3 = BasicBlock(128, 256, stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


# ===================== 训练 / 验证函数 =====================

def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        with autocast(enabled=(device == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.detach().item()
        running_loss += loss_val * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

            all_targets.append(y.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    return epoch_loss, epoch_acc, all_targets, all_preds


# ===================== 主流程 =====================

def main():
    set_seed(CFG["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    df = pd.read_csv(META_FILE)
    print("总样本数:", len(df))
    print(df.head(), "\n")
    print("SAMPLE NUMBER:", len(df))

    # 用 CSV 中的 class 列来确定类别顺序
    classes_list = sorted(df["class"].unique().tolist())
    print(classes_list)

    class_to_idx = {c: i for i, c in enumerate(classes_list)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    print(class_to_idx)

    # 训练 / 验证 / 测试划分
    train_ds = UrbanSoundMel3Ch(df, AUDIO_DIR, TRAIN_FOLDS, CFG,
                                class_to_idx, is_train=True)
    val_ds = UrbanSoundMel3Ch(df, AUDIO_DIR, VAL_FOLDS, CFG,
                              class_to_idx, is_train=False)
    test_ds = UrbanSoundMel3Ch(df, AUDIO_DIR, TEST_FOLDS, CFG,
                               class_to_idx, is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=CFG["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG["batch_size"],
        shuffle=False,
        num_workers=CFG["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=CFG["batch_size"],
        shuffle=False,
        num_workers=CFG["num_workers"],
        pin_memory=True,
    )

    print(
        f"Train: {len(train_ds)} samples | "
        f"Val: {len(val_ds)} | Test: {len(test_ds)}"
    )
    print(
        f" DataLoaders ready — "
        f"train:{len(train_loader)} batches, "
        f"val:{len(val_loader)}, test:{len(test_loader)}"
    )

    # 建模
    model = SmallResNet(n_classes=len(classes_list), in_channels=3).to(device)
    print(
        "模型参数量 (M):",
        sum(p.numel() for p in model.parameters()) / 1e6
    )

    # 类别权重（根据训练集频率）
    train_counts = train_ds.df["class"].value_counts().sort_index()
    max_count = train_counts.max()
    class_weights = (max_count / train_counts).values.astype(np.float32)
    print("class_weights:", class_weights)

    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG["lr"],
        weight_decay=CFG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = GradScaler(enabled=(device == "cuda"))

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    best_epoch = 0
    best_state = None
    patience = CFG["early_stop_patience"]

    # ========= 训练循环 =========
    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )
        val_loss, val_acc, _, _ = eval_model(
            model, val_loader, criterion, device
        )
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        dt = time.time() - t0
        print(
            f"[Epoch {epoch:02d}] "
            f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"time: {dt:.1f}s"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve = epoch - best_epoch

        if no_improve >= patience:
            print(
                f"Early stopping at epoch {epoch}, "
                f"best val acc = {best_acc:.4f} (epoch {best_epoch})"
            )
            break

    # ========= 使用最佳模型在测试集评估 =========
    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_path = "resnet3ch_best.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Best model saved to {ckpt_path}")
    test_loss, test_acc, y_true, y_pred = eval_model(
        model, test_loader, criterion, device
    )

    print("\n===== Test result (ResNet 3ch) =====")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    print("\nTest Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(classes_list))),
            target_names=classes_list,
            digits=3,
        )
    )

    # ========= 可视化：混淆矩阵 & 训练曲线 =========
    os.makedirs("plots", exist_ok=True)

    cm = confusion_matrix(
        y_true, y_pred,
        labels=list(range(len(classes_list)))
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes_list, yticklabels=classes_list
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("UrbanSound8K - Confusion Matrix (ResNet 3ch)")
    plt.tight_layout()
    plt.savefig("plots/confmat_resnet3ch_raw.png", dpi=300)
    plt.show()

    cm_norm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="YlGnBu",
        xticklabels=classes_list, yticklabels=classes_list
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("UrbanSound8K - Normalized Confusion Matrix (ResNet 3ch)")
    plt.tight_layout()
    plt.savefig("plots/confmat_resnet3ch_norm.png", dpi=300)
    plt.show()

    epochs_range = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss")
    plt.plot(epochs_range, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve (ResNet 3ch)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_acc"], label="Train Acc")
    plt.plot(epochs_range, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve (ResNet 3ch)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/training_curve_resnet3ch.png", dpi=300)
    plt.show()
    print("Training curves saved to plots/training_curve_resnet3ch.png")


# ===================== Windows 多进程安全入口 =====================

if __name__ == "__main__":
    main()
