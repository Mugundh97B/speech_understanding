import numpy as np
import scipy.fftpack as fft
import soundfile as sf
import matplotlib.pyplot as plt


# Load Audio

def load_audio(file_path):
    signal, sr = sf.read(file_path)
    return signal, sr


#  Pre-emphasis

def pre_emphasis(signal, alpha=0.97):
    emphasized = np.append(signal[0], signal[1:] - alpha * signal[:-1])
    return emphasized



#  Framing

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



#  Windowing

def apply_window(frames):
    return frames * np.hamming(frames.shape[1])



#  FFT & Power Spectrum

def power_spectrum(frames, NFFT=512):
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = (1.0 / NFFT) * (mag_frames ** 2)
    return pow_frames



#  Mel Filter Bank

def mel_filterbank(sr, NFFT, num_filters=40):
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sr / 2) / 700)

    mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filters + 2)
    hz_points = 700 * (10**(mel_points / 2595) - 1)

    bins = np.floor((NFFT + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((num_filters, int(NFFT / 2 + 1)))

    for m in range(1, num_filters + 1):
        f_m_minus = bins[m - 1]
        f_m = bins[m]
        f_m_plus = bins[m + 1]

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)

        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    return fbank



#  Apply Filterbank

def apply_filterbank(pow_frames, fbank):
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    return filter_banks



#  Log Compression

def log_compression(filter_banks):
    return np.log(filter_banks)



# DCT (MFCC)

def compute_mfcc(log_fbanks, num_ceps=13):
    mfcc = fft.dct(log_fbanks, type=2, axis=1, norm='ortho')[:, :num_ceps]
    return mfcc



# MAIN 

def extract_mfcc(file_path):
    signal, sr = load_audio(file_path)

    emphasized = pre_emphasis(signal)
    frames = framing(emphasized, sr)
    windowed = apply_window(frames)
    power_spec = power_spectrum(windowed)

    fbank = mel_filterbank(sr, 512)
    filter_banks = apply_filterbank(power_spec, fbank)

    log_fbanks = log_compression(filter_banks)
    mfcc = compute_mfcc(log_fbanks)

    return mfcc



# TEST 

if __name__ == "__main__":
    file_path = "data/sample.wav"  # place your audio file here

    mfcc = extract_mfcc(file_path)

    print("MFCC shape:", mfcc.shape)

    plt.imshow(mfcc.T, aspect='auto', origin='lower')
    plt.title("MFCC")
    plt.xlabel("Frames")
    plt.ylabel("Coefficients")
    plt.colorbar()
    plt.savefig("outputs/mfcc_plot.png", dpi=300, bbox_inches='tight')
    plt.show()