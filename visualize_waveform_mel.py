import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# ========= 配置 =========
URBANSOUND_ROOT = "data"
CSV_PATH = os.path.join(URBANSOUND_ROOT, "metadata", "UrbanSound8K.csv")
SAVE_DIR = "visualizations"

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


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def visualize_one(df, class_name):
    subset = df[df["class"] == class_name]
    if len(subset) == 0:
        print(f"[Warning] No audio found for class {class_name}")
        return

    row = subset.sample(1).iloc[0]
    fold = row["fold"]
    file = row["slice_file_name"]

    audio_path = os.path.join(
        URBANSOUND_ROOT, "build", "audio", f"fold{fold}", file
    )

    print(f"[{class_name}] → {audio_path}")

    y, sr = librosa.load(audio_path, sr=None, mono=True)

    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=64, fmax=8000
    )
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 6))

    ax1 = plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr, ax=ax1)
    ax1.set_title(f"Waveform - {class_name}")

    ax2 = plt.subplot(2, 1, 2)
    img = librosa.display.specshow(
        S_dB,
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="mel",
        fmax=8000,
        ax=ax2
    )
    ax2.set_title(f"Mel-Spectrogram - {class_name}")
    plt.colorbar(img, ax=ax2)

    out_path = os.path.join(SAVE_DIR, f"waveform_melspec_{class_name}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[Saved] {out_path}")


def main():
    ensure_dir(SAVE_DIR)

    df = pd.read_csv(CSV_PATH)
    print("Loaded metadata:", len(df))

    for cname in CLASS_NAMES:
        visualize_one(df, cname)

    print("\nAll visualizations saved in:", SAVE_DIR)


if __name__ == "__main__":
    main()
