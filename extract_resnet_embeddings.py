import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# 从训练脚本导入 Dataset / 模型 / 配置
from train_resnet_3ch import (
    UrbanSoundMel3Ch,
    SmallResNet,
    CLASS_NAMES,
    CFG,
    META_FILE,
    AUDIO_DIR,
    TRAIN_FOLDS,
    VAL_FOLDS,
    TEST_FOLDS,
)

CHECKPOINT_PATH = "resnet3ch_best.pth"
SAVE_NPZ_PATH = "resnet3ch_urbansound8k_embeddings.npz"
BATCH_SIZE = 64
NUM_WORKERS = 2


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}. "
            f"Please run train_resnet_3ch.py first to save the best model."
        )

    # 1) 准备数据：用全部 folds（1–10），但 is_train=False 只做确定性裁剪
    all_folds = TRAIN_FOLDS + VAL_FOLDS + TEST_FOLDS
    df = pd.read_csv(META_FILE)

    class_to_idx = {c: i for i, c in enumerate(sorted(df["class"].unique().tolist()))}

    ds_all = UrbanSoundMel3Ch(
        df=df,
        audio_dir=AUDIO_DIR,
        folds=all_folds,
        cfg=CFG,
        class_to_idx=class_to_idx,
        is_train=False,   # 固定裁剪方式，避免随机性
    )

    loader = DataLoader(
        ds_all,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Total samples for embedding extraction: {len(ds_all)}")

    # 2) 准备模型并加载最佳权重
    num_classes = len(CLASS_NAMES)
    model = SmallResNet(n_classes=num_classes, in_channels=3).to(device)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 3) 前向传播，取 FC 之前的 256 维特征
    all_feats = []
    all_labels = []

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            # 手动跑到 GAP + Dropout 之后，取 256 维 embedding
            h = model.stem(x)
            h = model.layer1(h)
            h = model.layer2(h)
            h = model.layer3(h)
            h = model.gap(h).view(h.size(0), -1)   # (B, 256)
            h = model.dropout(h)

            all_feats.append(h.cpu().numpy())
            all_labels.append(y.numpy())

    embeddings = np.concatenate(all_feats, axis=0)  # (N, 256)
    labels = np.concatenate(all_labels, axis=0)     # (N,)

    print("Embeddings shape:", embeddings.shape)
    print("Labels shape:", labels.shape)

    # 4) 保存为 .npz，格式和 YAMNet 一样 (embeddings + labels)
    np.savez(SAVE_NPZ_PATH, embeddings=embeddings, labels=labels)
    print(f"[Saved] {SAVE_NPZ_PATH}")


if __name__ == "__main__":
    main()
