import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# Load Data (same as Q2)

def load_data(data_dir="q2/librispeech/LibriSpeech/train-clean-100", n_samples=500):
    X = []
    y = []

    count = 0

    for speaker in os.listdir(data_dir):
        sp_path = os.path.join(data_dir, speaker)

        if not os.path.isdir(sp_path):
            continue

        for chapter in os.listdir(sp_path):
            ch_path = os.path.join(sp_path, chapter)

            for file in os.listdir(ch_path):
                if file.endswith(".flac"):
                    path = os.path.join(ch_path, file)

                    try:
                        audio, sr = librosa.load(path, sr=None)

                        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                        mfcc_mean = np.mean(mfcc, axis=1)

                        X.append(mfcc_mean)
                        y.append(speaker)

                        count += 1
                        if count >= n_samples:
                            return np.array(X), np.array(y)

                    except Exception:
                        continue

    return np.array(X), np.array(y)


# Prepare Data

def prepare_data():
    X, y = load_data()

    # 🔥 FAIRNESS STEP: normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    le = LabelEncoder()
    y = le.fit_transform(y)

    return train_test_split(X, y, test_size=0.2, random_state=42)



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


# Train

def train():
    X_train, X_test, y_train, y_test = prepare_data()

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    num_classes = int(max(y_train.max(), y_test.max())) + 1

    model = Model(X_train.shape[1], num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\nTraining FAIR model...")

    for epoch in range(20):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Evaluation
    with torch.no_grad():
        outputs = model(X_test)
        _, pred = torch.max(outputs, 1)
        acc = (pred == y_test).float().mean().item()

    print(f"\nFair Model Accuracy: {acc:.2f}")

    return acc



# MAIN
if __name__ == "__main__":
    acc = train()

    with open("q3/fair_results.txt", "w") as f:
        f.write(f"Fair Model Accuracy: {acc:.4f}\n")

    print("\nResults saved in q3/fair_results.txt")