import librosa
import soundfile as sf
import os
from privacymodule import apply_privacy



# Demo Function
def run_demo(data_dir="q2/librispeech/LibriSpeech/train-clean-100"):
    os.makedirs("q3/examples", exist_ok=True)

    print("Running Privacy Demo...")

    for speaker in os.listdir(data_dir):
        sp_path = os.path.join(data_dir, speaker)

        for chapter in os.listdir(sp_path):
            ch_path = os.path.join(sp_path, chapter)

            for file in os.listdir(ch_path):
                if file.endswith(".flac"):
                    path = os.path.join(ch_path, file)

                    audio, sr = librosa.load(path, sr=None)

                    private_audio = apply_privacy(audio, sr)

                    sf.write("q3/examples/demo_original.wav", audio, sr)
                    sf.write("q3/examples/demo_private.wav", private_audio, sr)

                    print("\nDemo completed!")
                    print("Check q3/examples/ folder")

                    return



# MAIN
if __name__ == "__main__":
    run_demo()