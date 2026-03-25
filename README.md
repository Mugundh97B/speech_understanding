# Speech Understanding System with Privacy & Fairness

##  Overview

This project implements a complete **Speech Understanding pipeline**, covering:

* Signal processing techniques (MFCC, spectral analysis)
* Speaker classification using machine learning
* Model improvement through feature engineering
* Ethical AI concepts: **bias detection, fairness, and privacy preservation**

The system is built using the **LibriSpeech dataset (train-clean-100)** and demonstrates how speech models can be improved and made more responsible.



##  Project Structure

```
speech_understanding/
├── mfcc_manual.py
├── leakage_snr.py
├── voiced_unvoiced.py
├── phonetic_mapping.py
├── q2/
│   ├── train.py
│   ├── eval.py
│   └── results/
├── q3/
│   ├── audit.py
│   ├── privacymodule.py
│   ├── train_fair.py
│   ├── pp_demo.py
│   └── examples/
├── requirements.txt
└── README.md
```


##  Q1: Speech Signal Processing

Implemented core speech processing techniques:

* **MFCC Extraction** (manual implementation)
* **Spectral Leakage & SNR Analysis**
* **Voiced vs Unvoiced Detection**
* **Phonetic Mapping (Wav2Vec2 model)**





##  Q2: Speaker Classification

###  Dataset

* **LibriSpeech – train-clean-100 (~6GB)**

###  Models Implemented

#### 1. Baseline Model

* Features: MFCC (13 coefficients)


#### 2. Improved Model

* Features:

  * MFCC
  * Energy
  * Zero Crossing Rate (ZCR)



 Output:

```
q2/results/
├── results.txt
└── comparison.png
```



##  Q3: Ethics, Fairness & Privacy

###  Bias Audit

* Checked speaker distribution
* Found imbalance in dataset

###  Privacy Module

* Added:

  * Noise injection
  * Pitch shifting
* Generated anonymized audio

###  Fair Model

* Applied feature normalization
* Accuracy improved to: **0.67**

###  Demo

* Original vs privacy-preserved audio comparison

Outputs:

```
q3/
├── audit_results.txt
├── fair_results.txt
└── examples/
    ├── demo_original.wav
    └── demo_private.wav
```



## Final Results Summary

| Model      | Accuracy |
| ---------- | -------- |
| Baseline   | 0.20     |
| Improved   | 0.47     |
| Fair Model | **0.67** |



##  Installation

```bash
git clone https://github.com/Mugundh97B/speech_understanding.git
cd speech_understanding

conda create -n speech_env python=3.10
conda activate speech_env

pip install -r requirements.txt
```



##  Run

### Q1

```bash
python mfcc_manual.py
python leakage_snr.py
python voiced_unvoiced.py
python phonetic_mapping.py
```

### Q2

```bash
python q2/train.py
python q2/eval.py
```

### Q3

```bash
python q3/audit.py
python q3/privacymodule.py
python q3/train_fair.py
python q3/pp_demo.py
```



## Dependencies

* numpy
* matplotlib
* librosa
* soundfile
* scikit-learn
* torch
* torchaudio
* transformers

##  Notes

* Large datasets (LibriSpeech) are excluded via `.gitignore`
* Only code and results are included in the repository
 

---


