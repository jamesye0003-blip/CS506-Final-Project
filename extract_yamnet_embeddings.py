import os
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm

# ------------ 配置 ------------
URBANSOUND_ROOT = "data"  # 改成你本地数据集根目录
CSV_PATH = os.path.join(URBANSOUND_ROOT, "metadata", "UrbanSound8K.csv")
OUTPUT_NPZ = "yamnet_urbansound8k_embeddings.npz"

TARGET_SR = 16000  # YAMNet 要求 16kHz

# ------------ 加载 YAMNet ------------
print("Loading YAMNet from TF Hub...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
print("YAMNet loaded.")

def load_and_resample(path, target_sr=TARGET_SR):
    """
    读取 wav 并重采样到 target_sr，返回 1D float32 waveform
    """
    wav, sr = sf.read(path)
    # 立体声 -> 单声道
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    # 重采样到 16k
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav.astype(np.float32)

def extract_yamnet_embedding(waveform):
    """
    waveform: 1D numpy array, 16kHz
    输出：单个 1024 维 embedding（时间维平均）
    """
    # YAMNet 需要 Tensor，shape: [num_samples]
    waveform_tf = tf.convert_to_tensor(waveform, dtype=tf.float32)
    # 模型输出：scores [num_patches, 521], embeddings [num_patches, 1024], spectrogram
    _, embeddings, _ = yamnet_model(waveform_tf)
    # 对时间维取平均，得到 [1024]
    emb = tf.reduce_mean(embeddings, axis=0)
    return emb.numpy()

def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Total rows in metadata: {len(df)}")

    embeddings = []
    labels = []
    folds = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        fold = row["fold"]
        class_id = int(row["classID"])
        file_name = row["slice_file_name"]
        fold_dir = f"fold{fold}"

        audio_path = os.path.join(URBANSOUND_ROOT, "build", "audio", fold_dir, file_name)
        if not os.path.exists(audio_path):
            print(f"Warning: file not found {audio_path}")
            continue

        try:
            wav = load_and_resample(audio_path)
            emb = extract_yamnet_embedding(wav)  # [1024]
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue

        embeddings.append(emb)
        labels.append(class_id)
        folds.append(fold)

    embeddings = np.stack(embeddings, axis=0)  # [N, 1024]
    labels = np.array(labels, dtype=np.int64)
    folds = np.array(folds, dtype=np.int64)

    print("Embeddings shape:", embeddings.shape)
    print("Saving to:", OUTPUT_NPZ)
    np.savez_compressed(
        OUTPUT_NPZ,
        embeddings=embeddings,
        labels=labels,
        folds=folds,
    )
    print("Done.")

if __name__ == "__main__":
    main()
