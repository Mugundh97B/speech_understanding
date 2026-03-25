import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os



# Load Audio
def load_audio(file_path):
    signal, sr = sf.read(file_path)
    return signal, sr



# Framing

def framing(signal, sr, frame_size=0.025, frame_stride=0.01):
    frame_length = int(frame_size * sr)
    frame_step = int(frame_stride * sr)

    signal_length = len(signal)
    num_frames = int(np.ceil((signal_length - frame_length) / frame_step)) + 1

    pad_length = num_frames * frame_step + frame_length
    pad_signal = np.append(signal, np.zeros(pad_length - signal_length))

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T

    frames = pad_signal[indices.astype(np.int32)]
    return frames



# Energy
def compute_energy(frames):
    return np.sum(frames ** 2, axis=1)



# Zero Crossing Rate
def compute_zcr(frames):
    zcr = np.sum(np.abs(np.diff(np.sign(frames))), axis=1) / (2 * frames.shape[1])
    return zcr



# Classification

def classify_frames(energy, zcr):
    energy_thresh = np.mean(energy)
    zcr_thresh = np.mean(zcr)

    labels = []

    for e, z in zip(energy, zcr):
        if e > energy_thresh and z < zcr_thresh:
            labels.append(1)  # voiced
        else:
            labels.append(0)  # unvoiced

    return np.array(labels)



# Main

def process(file_path):
    signal, sr = load_audio(file_path)

    frames = framing(signal, sr)

    energy = compute_energy(frames)
    zcr = compute_zcr(frames)

    labels = classify_frames(energy, zcr)

    os.makedirs("outputs", exist_ok=True)

    # Plot
    plt.figure(figsize=(12, 6))

    plt.subplot(3, 1, 1)
    plt.plot(signal)
    plt.title("Signal")

    plt.subplot(3, 1, 2)
    plt.plot(energy)
    plt.title("Energy")

    plt.subplot(3, 1, 3)
    plt.plot(labels)
    plt.title("Voiced (1) / Unvoiced (0)")

    plt.tight_layout()
    plt.savefig("outputs/voiced_unvoiced.png", dpi=300)
    plt.show()

    return labels



# RUN
if __name__ == "__main__":
    file_path = "data/sample.wav"

    labels = process(file_path)

    print("Voiced frames:", np.sum(labels))
    print("Unvoiced frames:", len(labels) - np.sum(labels))