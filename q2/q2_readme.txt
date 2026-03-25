Q2: Speaker Classification using LibriSpeech Dataset


1. Dataset Used

* Dataset: LibriSpeech (train-clean-100)

* Location:
  q2/librispeech/LibriSpeech/train-clean-100/

* Audio format: .flac

* Total samples used: 500 (subset for faster training)


2. Environment Setup

Create environment:

conda create -n speech_env python=3.10
conda activate speech_env

Install dependencies:

pip install -r requirements.txt



3.  to Run

Step 1: Train models

python q2/train.py

This will:

* Load LibriSpeech dataset
* Extract features (MFCC, energy, ZCR)
* Train two models:

  1. Baseline model
  2. Improved model
* Save results to:
  q2/results/results.txt


4. Evaluation

Run:

python q2/eval.py

This will:

* Load results
* Generate comparison plot:
  q2/results/comparison.png


5. Results (Checkpoints)

Results are stored in:
q2/results/results.txt

Observed results:

Baseline Accuracy: 0.20
Improved Accuracy: 0.47



6. Note

* Dataset folder is excluded from GitHub (.gitignore)
* Only code and results are tracked
* Subset of dataset used for faster execution
