import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



# Load LibriSpeech Dataset
def load_data(data_dir="q2/librispeech/LibriSpeech/train-clean-100", n_samples=500):
    X_base = []
    X_improved = []
    y = []

    print("Loading LibriSpeech dataset...")

    count = 0

    for speaker in os.listdir(data_dir):
        speaker_path = os.path.join(data_dir, speaker)

        if not os.path.isdir(speaker_path):
            continue

        for chapter in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter)

            for file in os.listdir(chapter_path):
                if file.endswith(".flac"):
                    path = os.path.join(chapter_path, file)

                    try:
                        audio, sr = librosa.load(path, sr=None)

                        if len(audio) < 1000:
                            continue

                        # Baseline
                        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                        mfcc_mean = np.mean(mfcc, axis=1)

                        # Improved 
                        energy = np.sum(audio ** 2)
                        zcr = np.mean(np.abs(np.diff(np.sign(audio))))

                        improved_features = np.concatenate([mfcc_mean, [energy, zcr]])

                        X_base.append(mfcc_mean)
                        X_improved.append(improved_features)
                        y.append(speaker)

                        count += 1

                        if count >= n_samples:
                            return np.array(X_base), np.array(X_improved), np.array(y)

                    except Exception:
                        continue

    return np.array(X_base), np.array(X_improved), np.array(y)



# Prepare Data
def prepare_data():
    X_base, X_improved, y = load_data()

    print("Total samples loaded:", len(y))

    le = LabelEncoder()
    y = le.fit_transform(y)

    Xb_train, Xb_test, y_train, y_test = train_test_split(
        X_base, y, test_size=0.2, random_state=42
    )

    Xi_train, Xi_test, _, _ = train_test_split(
        X_improved, y, test_size=0.2, random_state=42
    )

    return Xb_train, Xb_test, Xi_train, Xi_test, y_train, y_test



# Model

class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)



# Train Function

def train_model(X_train, y_train, X_test, y_test, input_size, name):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    num_classes = int(max(y_train.max(), y_test.max())) + 1

    model = Model(input_size, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"\nTraining {name} model...")

    for epoch in range(20):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"{name} Epoch {epoch}, Loss: {loss.item():.4f}")

    # Evaluation
    with torch.no_grad():
        outputs = model(X_test)
        _, pred = torch.max(outputs, 1)
        acc = (pred == y_test).float().mean().item()

    print(f"{name} Accuracy: {acc:.2f}")

    return acc



# MAIN

if __name__ == "__main__":
    os.makedirs("q2/results", exist_ok=True)

    Xb_train, Xb_test, Xi_train, Xi_test, y_train, y_test = prepare_data()

    # Baseline
    acc_base = train_model(Xb_train, y_train, Xb_test, y_test, 13, "Baseline")

    # Improved
    acc_improved = train_model(Xi_train, y_train, Xi_test, y_test, 15, "Improved")

    # Save results
    with open("q2/results/results.txt", "w") as f:
        f.write(f"Baseline Accuracy: {acc_base:.4f}\n")
        f.write(f"Improved Accuracy: {acc_improved:.4f}\n")

    print("\nResults saved in q2/results/")