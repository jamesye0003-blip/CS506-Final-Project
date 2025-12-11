import os
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm


URBANSOUND_ROOT = "data"
CSV_PATH = os.path.join(URBANSOUND_ROOT, "metadata", "UrbanSound8K.csv")
OUTPUT_NPZ = "yamnet_urbansound8k_embeddings.npz"

TARGET_SR = 16000


print("Loading YAMNet from TF Hub...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
print("YAMNet loaded.")
#Load a WAV file and resample it to target_sr.
def load_and_resample(path, target_sr=TARGET_SR):

    wav, sr = sf.read(path)
    # Stereo → mono by averaging channels
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    # Resample to 16 kHz if needed
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav.astype(np.float32)

#Extract a single 1024-dimensional YAMNet embedding from a 1D waveform.
def extract_yamnet_embedding(waveform):

    # YAMNet expects a Tensor of shape
    waveform_tf = tf.convert_to_tensor(waveform, dtype=tf.float32)
    # Model outputs
    _, embeddings, _ = yamnet_model(waveform_tf)
    # Average across time patches
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
