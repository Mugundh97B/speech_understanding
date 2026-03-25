import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os



# Load Audio

def load_audio(file_path):
    signal, sr = sf.read(file_path)
    return signal, sr



# Window Functions

def get_windows(N):
    return {
        "Rectangular": np.ones(N),
        "Hamming": np.hamming(N),
        "Hanning": np.hanning(N)
    }


# Spectral Leakage Measure

def spectral_leakage(spectrum):
    main_lobe = np.max(spectrum)
    leakage = np.sum(spectrum) - main_lobe
    return leakage



# SNR Calculation

def compute_snr(signal):
    signal_power = np.mean(signal ** 2)
    noise_power = np.var(signal - np.mean(signal))
    return 10 * np.log10(signal_power / (noise_power + 1e-10))  



# Main Analysis

def analyze(file_path):
    signal, sr = load_audio(file_path)

    # Take small segment
    segment = signal[:1024]

    windows = get_windows(len(segment))

    results = {}

    os.makedirs("outputs", exist_ok=True)

    for name, window in windows.items():
        windowed = segment * window

        spectrum = np.abs(np.fft.fft(windowed))

        leakage = spectral_leakage(spectrum)
        snr = compute_snr(windowed)

        results[name] = {"leakage": leakage, "snr": snr}

        # Plot spectrum
        plt.plot(spectrum, label=name)

    # Save plot
    plt.title("Spectral Leakage Comparison")
    plt.legend()
    plt.savefig("outputs/leakage_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

    return results



# RUN

if __name__ == "__main__":
    file_path = "data/sample.wav"

    results = analyze(file_path)

    print("\nComparison Results:")
    for k, v in results.items():
        print(f"{k}: Leakage={v['leakage']:.2f}, SNR={v['snr']:.2f}")