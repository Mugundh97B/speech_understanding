import numpy as np
import librosa



# Add Noise

def add_noise(audio, noise_level=0.02):
    noise = np.random.randn(len(audio))
    return audio + noise_level * noise



# Pitch Shift
def pitch_shift(audio, sr, steps=2):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)



# Privacy Transform
def apply_privacy(audio, sr):
    audio_noisy = add_noise(audio)
    audio_shifted = pitch_shift(audio_noisy, sr)

    return audio_shifted



# TEST
if __name__ == "__main__":
    import soundfile as sf
    import os

    os.makedirs("q3/examples", exist_ok=True)

    file_path = "q2/librispeech/LibriSpeech/train-clean-100"


    for speaker in os.listdir(file_path):
        sp_path = os.path.join(file_path, speaker)

        for chapter in os.listdir(sp_path):
            ch_path = os.path.join(sp_path, chapter)

            for file in os.listdir(ch_path):
                if file.endswith(".flac"):
                    full_path = os.path.join(ch_path, file)

                    audio, sr = librosa.load(full_path, sr=None)

                    private_audio = apply_privacy(audio, sr)

                    sf.write("q3/examples/original.wav", audio, sr)
                    sf.write("q3/examples/private.wav", private_audio, sr)

                    print("Saved original.wav and private.wav")
                    exit()