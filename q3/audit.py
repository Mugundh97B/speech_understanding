import os
from collections import Counter
import matplotlib.pyplot as plt


# Audit Dataset

def audit_dataset(data_dir="q2/librispeech/LibriSpeech/train-clean-100"):
    speaker_counts = Counter()

    for speaker in os.listdir(data_dir):
        speaker_path = os.path.join(data_dir, speaker)

        if not os.path.isdir(speaker_path):
            continue

        count = 0

        for chapter in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter)

            for file in os.listdir(chapter_path):
                if file.endswith(".flac"):
                    count += 1

        speaker_counts[speaker] = count

    return speaker_counts



# MAIN
if __name__ == "__main__":
    counts = audit_dataset()

    print("\nSpeaker Distribution:\n")

    for speaker, count in list(counts.items())[:10]:
        print(f"Speaker {speaker}: {count} samples")

    print("\nTotal Speakers:", len(counts))

    # results
    with open("q3/audit_results.txt", "w") as f:
        for speaker, count in counts.items():
            f.write(f"{speaker}: {count}\n")

    print("\nResults saved in q3/audit_results.txt")

if __name__ == "__main__":
    counts = audit_dataset()

    speakers = list(counts.keys())
    values = list(counts.values())

    print("Total Speakers:", len(speakers))

    # Plot
    plt.figure(figsize=(10, 5))
    plt.hist(values, bins=20)
    plt.xlabel("Samples per Speaker")
    plt.ylabel("Frequency")
    plt.title("Speaker Distribution")

    plt.savefig("q3/audit_plots.pdf")
    plt.show()

    print("Saved audit_plots.pdf")