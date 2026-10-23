# Real-Time Multimodal Emotion Analytics

A real-time system for detecting human emotions by combining facial expression analysis and speech prosody based on my undergradaute thesis paper
**"Towards Real-Time Emotion Analytics: Integrating Facial Landmarks and Speech Prosody".**

The system processes live webcam video and microphone audio, runs two independent neural network models, and fuses their predictions to produce a stable emotion label with confidence score.

**Supported emotions:** Angry · Fear · Happy · Sad · Neutral

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Models](#models)
- [Fusion Strategy](#fusion-strategy)
- [Real-Time Pipeline](#real-time-pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [License](#license)

---

## Overview

This project implements a complete multimodal emotion recognition pipeline designed for real-time use. It consists of three main stages:

1. **Video branch** – Detects faces in the webcam stream using MediaPipe, crops and preprocesses the face region, and classifies the expression with a MobileNetV2-based network.
2. **Audio branch** – Continuously records audio from the microphone, extracts MFCC features (including first and second derivatives), and classifies the emotional content of speech with a Bidirectional LSTM.
3. **Fusion module** – Combines the probability distributions from both modalities using a weighted average, applies temporal smoothing, and outputs a final emotion label together with a confidence score.

The design prioritizes modularity, reproducibility, and low-latency inference suitable for interactive applications.

---

## System Architecture

```
┌──────────────────┐
│     Webcam       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     ┌─────────────────────┐
│ MediaPipe Face   │────▶│  MobileNetV2        │──▶ video probabilities (5)
│ Detection        │     │  (48×48 input)      │
└──────────────────┘     └─────────────────────┘
                                      │
┌──────────────────┐                  │
│   Microphone     │                  │
└────────┬─────────┘                  │
         │                            │
         ▼                            │
┌──────────────────┐     ┌─────────────────────┐
│ MFCC + Δ + ΔΔ    │────▶│  Bidirectional LSTM │──▶ audio probabilities (5)
│ Feature Extract  │     │                     │
└──────────────────┘     └─────────────────────┘
                                      │
                                      ▼
                         ┌─────────────────────┐
                         │   Late Fusion       │
                         │ • Weighted average  │
                         │ • Temporal smoothing│
                         └──────────┬──────────┘
                                    │
                                    ▼
                             Final Emotion
                           + Confidence Score
```

---

## Project Structure

```
emotion_analytics/
├── data/
│   ├── raw/                  # Original dataset files
│   ├── processed/            # Preprocessed .npz files ready for training
│   ├── fer2013/
│   └── ravdess/
├── models/                   # Trained model weights (.keras)
├── src/
│   ├── config.py             # Central configuration (paths, hyperparameters, labels)
│   ├── models/
│   │   ├── video_model.py    # MobileNetV2-based facial emotion classifier
│   │   └── audio_model.py    # Bidirectional LSTM for speech emotion
│   ├── pipeline/
│   │   ├── fusion.py         # Weighted late fusion + temporal smoothing
│   │   └── realtime.py       # Live webcam + microphone demo
│   └── utils/
│       ├── video_utils.py    # MediaPipe detection and face preprocessing
│       ├── audio_utils.py    # MFCC extraction and audio loading
│       └── common.py         # Seed setting, directory helpers
├── scripts/
│   ├── prepare_fer2013.py    # Download & preprocess FER-2013
│   ├── prepare_ravdess.py    # Extract features from RAVDESS
│   ├── train_video.py        # Train the facial model
│   ├── train_audio.py        # Train the speech model
│   ├── evaluate.py           # Test-set evaluation and confusion matrices
│   └── run_realtime.py       # Launch the real-time demo
├── docs/                     # Additional design documentation
├── requirements.txt
└── README.md
```

---

## Datasets

### Facial Expression – FER-2013

- Source: [Kaggle – FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)
- Original 7 emotion classes filtered to the 5 target emotions
- Images are 48×48 grayscale
- Preprocessing pipeline converts pixel strings into normalized 48×48×3 tensors (grayscale channel replicated) and creates stratified train / validation / test splits

### Speech Emotion – RAVDESS

- Source: [Zenodo – RAVDESS](https://zenodo.org/record/1188976) (Audio_Speech_Actors_01-24)
- Only the speech portion is used
- Emotion codes are mapped to the same 5 labels
- Features extracted: 40 MFCCs + first-order delta + second-order delta, padded/truncated to a fixed length of 130 frames

Both datasets are processed into compressed `.npz` files stored under `data/processed/` for fast loading during training.

---

## Models

### Video Model (`src/models/video_model.py`)

| Component          | Details                                       |
| ------------------ | --------------------------------------------- |
| Backbone           | MobileNetV2 (ImageNet pretrained)             |
| Input shape        | (48, 48, 3)                                   |
| Pooling            | Global Average Pooling                        |
| Head               | Dense(256) → BatchNorm → Dropout → Softmax(5) |
| Loss               | Sparse Categorical Cross-Entropy              |
| Optimizer          | Adam                                          |
| Trainable backbone | Frozen by default (can be unfrozen later)     |

### Audio Model (`src/models/audio_model.py`)

| Component      | Details                                                      |
| -------------- | ------------------------------------------------------------ |
| Input shape    | (130, 120) → 40 MFCC + Δ + ΔΔ                                |
| Architecture   | Masking → BiLSTM(128) → BiLSTM(64) → Dense(128) → Softmax(5) |
| Regularization | Dropout + Recurrent Dropout                                  |
| Loss           | Sparse Categorical Cross-Entropy                             |
| Optimizer      | Adam                                                         |

---

## Fusion Strategy

The system uses **late fusion**:

1. Each modality independently produces a 5-dimensional probability vector (or `None` if the modality is unavailable).
2. A weighted average is computed:
   ```
   fused = w_video * video_probs + w_audio * audio_probs
   ```
   Default weights: `0.45` (video) / `0.55` (audio).
3. Temporal smoothing is applied by averaging the last _N_ fused probability vectors (default window = 7).
4. The emotion with the highest probability is selected. If the confidence falls below a threshold, the system outputs `"Uncertain"`.
5. Graceful degradation is supported: missing face → pure audio; silent/noisy audio → pure video.

Implementation: `src/pipeline/fusion.py`

---

## Real-Time Pipeline

- **Face detection**: MediaPipe Face Detection (short-range model)
- **Face preprocessing**: crop with margin → grayscale → resize 48×48 → replicate to 3 channels → normalize
- **Audio capture**: non-blocking PyAudio stream with a circular buffer
- **Inference schedule**: video every frame, audio approximately every 0.5 seconds
- **Display**: bounding box, emotion label, confidence percentage, optional FPS counter
- **Controls**: `q` = quit, `r` = reset fusion history

Entry point: `scripts/run_realtime.py` or `python -m src.pipeline.realtime`

---

## Installation

### 1. Clone / Extract

```bash
cd emotion_analytics
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. System Dependencies

| Platform      | Command / Action                                      |
| ------------- | ----------------------------------------------------- |
| Ubuntu/Debian | `sudo apt install portaudio19-dev ffmpeg libsndfile1` |
| macOS         | `brew install portaudio ffmpeg`                       |
| Windows       | Install Microsoft C++ Build Tools if PyAudio fails    |

---

## Usage

### Prepare Datasets

```bash
# Place fer2013.csv in data/raw/
python scripts/prepare_fer2013.py

# Extract RAVDESS so that data/raw/ravdess/Actor_* exists
python scripts/prepare_ravdess.py
```

### Train Models

```bash
python scripts/train_video.py
python scripts/train_audio.py
```

Trained weights are saved to the `models/` directory.

### Evaluate (optional)

```bash
python scripts/evaluate.py
```

### Run Real-Time Demo

```bash
python scripts/run_realtime.py
```

---

## Configuration

All important constants live in `src/config.py`:

- Emotion labels and index mappings
- Image size (48×48)
- Audio parameters (sample rate, number of MFCCs, padding length)
- Fusion weights and temporal window size
- Training hyperparameters (batch size, learning rate, epochs)
- Model filenames and paths

Change values in this single file to experiment with different settings.

---

## Dependencies

Core libraries:

- TensorFlow / Keras ≥ 2.15
- OpenCV
- MediaPipe
- librosa + soundfile
- PyAudio
- NumPy, Pandas, scikit-learn
- Matplotlib, Seaborn, tqdm

Full list with version constraints is available in `requirements.txt`.
