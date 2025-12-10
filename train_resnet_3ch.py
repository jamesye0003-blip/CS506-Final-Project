import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

import librosa
import librosa.display

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# ===================== 基本配置 =====================

DATA_ROOT = Path("data")
AUDIO_DIR = DATA_ROOT / "build" / "audio"
META_FILE = DATA_ROOT / "metadata" / "UrbanSound8K.csv"

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

CFG = {
    "epochs": 80,
    "batch_size": 32,
    "lr": 1e-4,
    "weight_decay": 3e-4,
    "sample_rate": None,      # 不强制重采样
    "n_mels": 64,
    "n_fft": 1024,
    "fixed_seconds": 4.0,
    "hop_length": 512,
    "fmax": 8000,
    "num_workers": 2,
    "seed": 42,
    "mixup_alpha": 0.4,
    "T_fixed": None,          # 由 frames_for_seconds 计算
    "early_stop_patience": 10,
}

TRAIN_FOLDS = [1, 2, 3, 4, 5, 6, 7, 8]
VAL_FOLDS   = [9]
TEST_FOLDS  = [10]


# ===================== 工具函数 =====================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def frames_for_seconds(seconds, hop_length, sr):
    return int(np.ceil(seconds * sr / hop_length))


def wav_to_logmel(y, sr, n_mels, n_fft, hop_length, fmax):
    """和 notebook 一致的 log-mel 计算"""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmax=fmax,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)  # (n_mels, T)


def pad_or_crop_time(x, T, is_train=True):
    """
    x: (n_mels, T_orig)
    T: 固定帧数
    训练: 随机裁剪 / 右侧补 0
    验证/测试: 中心裁剪 / 右侧补 0
    """
    n_mels, T_orig = x.shape
    if T_orig == T:
        return x

    if T_orig > T:
        # 裁剪
        if is_train:
            start = np.random.randint(0, T_orig - T + 1)
        else:
            start = max(0, (T_orig - T) // 2)
        return x[:, start:start + T]

    # T_orig < T → 右侧补 0
    pad = T - T_orig
    return np.pad(x, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)


def spec_augment(mel, max_mask_freq=12, max_mask_time=20):
    """
    简化版 SpecAugment:
    - 随机频率遮挡
    - 随机时间遮挡
    只在训练集上使用
    """
    mel = mel.copy()
    n_mels, T = mel.shape

    # freq mask
    f = np.random.randint(0, max_mask_freq + 1)
    if f > 0:
        f0 = np.random.randint(0, max(1, n_mels - f + 1))
        mel[f0:f0 + f, :] = 0.0

    # time mask
    t = np.random.randint(0, max_mask_time + 1)
    if t > 0:
        t0 = np.random.randint(0, max(1, T - t + 1))
        mel[:, t0:t0 + t] = 0.0

    return mel


# ===================== Dataset：完全对齐 notebook 版 =====================

class UrbanSoundMel3Ch(Dataset):
    def __init__(self, df, audio_dir, folds, cfg, class_to_idx, is_train: bool):
        self.df = df[df["fold"].isin(folds)].reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.cfg = cfg
        self.class_to_idx = class_to_idx
        self.n_mels = cfg["n_mels"]
        self.is_train = is_train

        # 共享 T_fixed：和 notebook 一样，按 sr=48000 估算
        if cfg["T_fixed"] is None:
            cfg["T_fixed"] = frames_for_seconds(
                cfg["fixed_seconds"], cfg["hop_length"], 48000
            )
        self.T_fixed = cfg["T_fixed"]
        print(f"[UrbanSoundMel3Ch][{'train' if is_train else 'val/test'}] T_fixed={self.T_fixed}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        wav_path = self.audio_dir / f"fold{row.fold}" / row.slice_file_name

        # ---- 读音频（和 notebook 一样用 soundfile）----
        y, sr = sf.read(wav_path)

        # 可选：如果担心 sr 太低，可以强制重采样到某个 sample_rate
        # if self.cfg["sample_rate"]:
        #     if sr != self.cfg["sample_rate"]:
        #         y = librosa.resample(y, orig_sr=sr, target_sr=self.cfg["sample_rate"])
        #         sr = self.cfg["sample_rate"]

        if y.ndim == 2:  # 立体声 → 单声道
            y = y.mean(axis=1)

        # ---- Mel ----
        mel = wav_to_logmel(
            y, sr,
            self.cfg["n_mels"], self.cfg["n_fft"],
            self.cfg["hop_length"], self.cfg["fmax"]
        )  # (n_mels, T_orig)

        # 先固定时间长度，再做 delta / delta2
        mel = pad_or_crop_time(mel, self.T_fixed, is_train=self.is_train)

        # 只在训练集上做 SpecAugment
        if self.is_train:
            mel = spec_augment(mel)

        # 这里 mel 长度已经是 T_fixed，远大于 9，delta 不会再报 width 问题
        delta = librosa.feature.delta(mel, mode="nearest")
        delta2 = librosa.feature.delta(mel, order=2, mode="nearest")

        # ---- 归一化到 [0,1]（和 notebook 一样）----
        def norm01(x):
            mn, mx = x.min(), x.max()
            return (x - mn) / (mx - mn + 1e-6)

        mel = norm01(mel)
        delta = norm01(delta)
        delta2 = norm01(delta2)

        x = np.stack([mel, delta, delta2], axis=0)  # (3, n_mels, T_fixed)
        x = torch.tensor(x, dtype=torch.float32)

        y_label = torch.tensor(self.class_to_idx[row["class"]], dtype=torch.long)
        return x, y_label, str(wav_path)


# ===================== 小 ResNet 模型 =====================

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(identity)
        out = self.relu(out)
        return out


class SmallResNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = BasicBlock(32, 64, stride=2)
        self.layer2 = BasicBlock(64, 128, stride=2)
        self.layer3 = BasicBlock(128, 256, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


# ===================== 训练 / 验证 =====================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_true, all_pred = [], []

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += x.size(0)

            all_true.extend(y.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())

    return total_loss / total, correct / total, np.array(all_true), np.array(all_pred)


# ===================== 主入口 =====================

def main():
    set_seed(CFG["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    df = pd.read_csv(META_FILE)
    print("SAMPLE NUMBER:", len(df))

    classes = sorted(df["class"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    print("Classes:", classes)
    print("class_to_idx:", class_to_idx)

    train_ds = UrbanSoundMel3Ch(df, AUDIO_DIR, TRAIN_FOLDS, CFG, class_to_idx, is_train=True)
    val_ds   = UrbanSoundMel3Ch(df, AUDIO_DIR, VAL_FOLDS, CFG, class_to_idx, is_train=False)
    test_ds  = UrbanSoundMel3Ch(df, AUDIO_DIR, TEST_FOLDS, CFG, class_to_idx, is_train=False)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

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

    model = SmallResNet(in_channels=3, n_classes=len(classes)).to(device)
    print("模型参数量 (M):", sum(p.numel() for p in model.parameters()) / 1e6)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"]
    )

    best_val_acc = 0.0
    best_state = None
    no_improve = 0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, CFG["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = eval_model(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"[Epoch {epoch:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve} epoch(s).")

        if no_improve >= CFG["early_stop_patience"]:
            print(f"\nEarly stopping at epoch {epoch}. Best val acc = {best_val_acc:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, y_true, y_pred = eval_model(model, test_loader, criterion, device)
    print("\n===== Test result (ResNet 3ch) =====")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    print("\nTest Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(classes))),
            target_names=classes,
            digits=3,
        )
    )

    os.makedirs("plots", exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("UrbanSound8K - Normalized Confusion Matrix (ResNet 3ch)")
    plt.tight_layout()
    cm_path = "plots/confmat_resnet3ch.png"
    plt.savefig(cm_path, dpi=300)
    plt.show()
    print("Confusion matrix figure saved to", cm_path)

    # 训练曲线
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
    curve_path = "plots/training_curve_resnet3ch.png"
    plt.savefig(curve_path, dpi=300)
    plt.show()
    print("Training curve saved to", curve_path)


if __name__ == "__main__":
    main()
