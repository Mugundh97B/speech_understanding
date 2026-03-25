import librosa
import numpy as np


def compute_snr(original, modified):
    noise = original - modified
    snr = 10 * np.log10(np.sum(original**2) / np.sum(noise**2))
    return snr


def evaluate():
    orig_path = "q3/examples/demo_original.wav"
    priv_path = "q3/examples/demo_private.wav"

    original, sr = librosa.load(orig_path, sr=None)
    modified, _ = librosa.load(priv_path, sr=None)

    min_len = min(len(original), len(modified))
    original = original[:min_len]
    modified = modified[:min_len]

    snr = compute_snr(original, modified)

    print(f"SNR (Privacy Distortion): {snr:.2f} dB")

    with open("q3/evaluation_scripts/results.txt", "w") as f:
        f.write(f"SNR: {snr:.2f} dB\n")


if __name__ == "__main__":
    evaluate()