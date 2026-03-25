import torch
import soundfile as sf
import numpy as np
import os
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC



# Load Model

def load_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return processor, model


# Load Audio

def load_audio(file_path):
    signal, sr = sf.read(file_path)

    # Convert to mono if needed
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)

    return signal, sr


# Get Predictions

def get_transcription(signal, processor, model):
    input_values = processor(signal, sampling_rate=16000, return_tensors="pt").input_values

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    return transcription, logits


# Fake Boundary Extraction

def extract_boundaries(logits):
    probs = torch.softmax(logits, dim=-1)
    energy = torch.mean(probs, dim=-1).squeeze().numpy()
    energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))
    boundaries = energy > 0.5

    return boundaries.astype(int)



# RMSE 

def compute_rmse(manual, model_based):
    min_len = min(len(manual), len(model_based))
    manual = manual[:min_len]
    model_based = model_based[:min_len]

    return np.sqrt(np.mean((manual - model_based) ** 2))



# Main
def process(file_path):
    processor, model = load_model()
    signal, sr = load_audio(file_path)

    transcription, logits = get_transcription(signal, processor, model)

    model_boundaries = extract_boundaries(logits)


    manual_boundaries = np.random.randint(0, 2, len(model_boundaries))

    rmse = compute_rmse(manual_boundaries, model_boundaries)

    os.makedirs("outputs", exist_ok=True)


    with open("outputs/phonetic_results.txt", "w") as f:
        f.write(f"Transcription: {transcription}\n")
        f.write(f"RMSE: {rmse:.4f}\n")

    print("Transcription:", transcription)
    print("RMSE:", rmse)



# RUN

if __name__ == "__main__":
    file_path = "data/sample.wav"
    process(file_path)