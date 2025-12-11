import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import Dataset / model / config from the training script
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

    # 1) Prepare data: use all folds (1–10)
    all_folds = TRAIN_FOLDS + VAL_FOLDS + TEST_FOLDS
    df = pd.read_csv(META_FILE)

    class_to_idx = {c: i for i, c in enumerate(sorted(df["class"].unique().tolist()))}

    ds_all = UrbanSoundMel3Ch(
        df=df,
        audio_dir=AUDIO_DIR,
        folds=all_folds,
        cfg=CFG,
        class_to_idx=class_to_idx,
        is_train=False,   # fixed cropping
    )

    loader = DataLoader(
        ds_all,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Total samples for embedding extraction: {len(ds_all)}")

    # 2) Build model and load best checkpoint
    num_classes = len(CLASS_NAMES)
    model = SmallResNet(n_classes=num_classes, in_channels=3).to(device)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 3) Forward pass and extract 256-dim features before the FC layer
    all_feats = []
    all_labels = []

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            # Manually run up to GAP + Dropout and take 256-dim embeddings
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

    # 4) Save as
    np.savez(SAVE_NPZ_PATH, embeddings=embeddings, labels=labels)
    print(f"[Saved] {SAVE_NPZ_PATH}")


if __name__ == "__main__":
    main()
