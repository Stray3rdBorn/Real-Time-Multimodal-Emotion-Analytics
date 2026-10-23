# Architecture, Design & Planning Document

**Real-Time Multimodal Emotion Analytics**

---

## 1. Project Goals

1. Reconstruct a fully functional multimodal emotion recognition system based on my undergraduate thesis paper on **"Towards Real-Time Emotion Analytics: Integrating Facial Landmarks and Speech Prosody".**
2. Replace all private/custom data with **public, widely-used datasets**.
3. Apply modern engineering and modelling improvements while keeping the system fast enough for real-time webcam + microphone use.
4. Produce clean, modular, well-documented code that can be expanded in the future.

---

## 2. Emotion Taxonomy

Exactly the five emotions used in the thesis:

| Index | Emotion |
| ----- | ------- |
| 0     | Angry   |
| 1     | Fear    |
| 2     | Happy   |
| 3     | Sad     |
| 4     | Neutral |

Disgust and Surprise (present in FER-2013 and RAVDESS) are deliberately dropped to keep the output space identical to the thesis.

---

## 3. Datasets

### 3.1 Video – FER-2013

- Source: Kaggle (`msambare/fer2013`)
- Original 7 classes → filtered to the 5 classes above
- Images already 48×48 grayscale → perfect match for the original pipeline resolution
- Processed into stratified train / val / test splits and stored as `data/processed/fer2013_5emo.npz`

### 3.2 Audio – RAVDESS

- Source: Zenodo (Audio_Speech_Actors_01-24)
- Only speech portion used
- Emotion codes mapped to the same 5 labels
- Features extracted once and cached as `data/processed/ravdess_5emo.npz`

**Why these two?**

- FER-2013 is the de-facto benchmark for facial expression recognition at 48×48.
- RAVDESS is the dataset most frequently referenced in the original thesis and provides high-quality acted emotional speech.
- Both are free for research and fully public → perfect for reproducibility and interviews.

---

## 4. System Architecture

### 4.1 High-level flow (late fusion)

```
Webcam frame ──▶ MediaPipe Face Detection ──▶ 48×48 face ──▶ MobileNetV2 ──▶ video_probs
                                                                                 │
Microphone ────▶ MFCC+Δ+ΔΔ (130×120) ──────▶ BiLSTM ──────────────────────────▶ audio_probs
                                                                                 │
                                                                   Weighted average
                                                                   + temporal smoothing
                                                                                 │
                                                                          Final label
```

### 4.2 Video Branch

- **Detection**: MediaPipe Face Detection (short-range model). Far more robust than the Haar cascades used in the original work.
- **Preprocessing**: Crop with small margin → grayscale → resize 48×48 → replicate to 3 channels → [0,1] → MobileNetV2 preprocess_input.
- **Model**: MobileNetV2 (ImageNet weights) + global average pooling + Dropout + Dense(256) + BatchNorm + Softmax(5).
- Backbone is frozen by default; can be unfrozen for fine-tuning later.
- Input size deliberately kept at 48×48 to honour the original real-time constraint and to allow direct comparison with the thesis baseline.

### 4.3 Audio Branch

- **Features**: 40 MFCCs + first-order delta + second-order delta → 120-dimensional feature vector per frame.
- Fixed length 130 frames (≈ 3 s at the chosen hop length) with zero-padding / truncation.
- **Model**: Masking → BiLSTM(128) → BiLSTM(64) → Dense(128) → Softmax(5).
- Bidirectional LSTMs capture both past and future context inside the short utterance, improving over the unidirectional LSTM of the original notebook.

### 4.4 Fusion Module

1. Each modality produces a 5-dimensional probability vector (or `None` if the modality is unavailable).
2. **Weighted average**:  
   `fused = w_v * video_probs + w_a * audio_probs`  
   Default weights: `(0.45, 0.55)` – slight preference for audio because speech prosody is often more reliable under varying lighting.
3. **Temporal smoothing**: moving average of the last 7 fused vectors. This dramatically reduces label flicker.
4. Confidence threshold (default 0.35). Below the threshold the system outputs “Uncertain”.
5. Graceful degradation: if face is lost → pure audio; if audio is silent → pure video.

### 4.5 Real-time Pipeline

- OpenCV webcam capture (mirrored).
- Non-blocking PyAudio stream that continuously fills a circular buffer.
- Face detection + video inference every frame.
- Audio inference every ~8 frames (≈ 0.5 s) to keep CPU load reasonable.
- Overlay: bounding box, emotion label, confidence percentage, optional FPS.
- Keyboard controls: `q` quit, `r` reset fusion history.

---

## 5. Training Protocol

- Optimizer: Adam (lr = 1e-4)
- Loss: Sparse categorical cross-entropy
- Callbacks: EarlyStopping (patience 8–10), ReduceLROnPlateau, ModelCheckpoint (best val accuracy), TensorBoard
- Class balance: stratified splits; class weights can be added later if needed
- Data augmentation (video): can be enabled via `tf.keras.preprocessing.image.ImageDataGenerator` or Albumentations in a future iteration

---

## 6. Implementation Notes & Trade-offs

| Decision                        | Rationale                                                                   |
| ------------------------------- | --------------------------------------------------------------------------- |
| Keep 48×48                      | Honour original real-time target; avoid large compute jump at first version |
| MobileNetV2 instead of tiny CNN | Immediate accuracy boost + transfer learning with almost no extra latency   |
| MediaPipe over Haar             | Far better recall under non-frontal faces and varied lighting               |
| Separate models + late fusion   | Matches thesis, easier to train, easy to replace one modality               |
| Public datasets only            | Full reproducibility; no “I lost the data” problem in interviews            |
| Modular package structure       | Professional, testable, GitHub-ready                                        |

---

## 7. How the System Relates to the Original Thesis

- Same five emotion classes.
- Same high-level pipeline: face → CNN-style model, speech → LSTM on prosodic features, late fusion, real-time demo.
- Same resolution (48×48).
- Improvements are additive and clearly labelled as post-thesis engineering upgrades.
- The original custom video dataset is replaced by FER-2013 so that every number can be reproduced by anyone.

---

## 8. Known Limitations

1. Both FER-2013 and RAVDESS are **acted** / posed data. Performance on spontaneous in-the-wild video will be lower.
2. Cultural and demographic bias exists in both datasets.
3. 48×48 discards fine-grained texture that larger models exploit.
4. Late fusion cannot model cross-modal interactions as richly as early or attention-based fusion.
5. Real-time audio window is short (≈1.5 s); longer context can help some emotions (e.g. sadness).

---

## 9. Suggested Future Work

- Add MediaPipe Face Mesh landmarks as additional features or as a parallel branch.
- Experiment with early fusion or cross-attention transformers.
- Move to 96×96 or 112×112 once the 48×48 baseline is solid.
- Export to TFLite / ONNX and measure on-device latency.
- Collect a small personal test set for qualitative evaluation.
- Add Gradio / Streamlit web interface for easy demos.

---

## 10. File & Script Reference

| Script / Module                   | Purpose                                  |
| --------------------------------- | ---------------------------------------- |
| `scripts/prepare_fer2013.py`      | Download + filter + split FER-2013       |
| `scripts/prepare_ravdess.py`      | Extract MFCC features from RAVDESS       |
| `scripts/train_video.py`          | Train MobileNetV2                        |
| `scripts/train_audio.py`          | Train BiLSTM                             |
| `scripts/run_realtime.py`         | Launch live demo                         |
| `src/config.py`                   | Single source of truth for all constants |
| `src/pipeline/fusion.py`          | Weighted + temporal fusion logic         |
| `src/pipeline/realtime.py`        | Full real-time loop                      |
| `docs/ARCHITECTURE_AND_DESIGN.md` | This document                            |

---

_Document version: 1.0 – October 2026_  
_Corresponds to the modern public-dataset reconstruction of the original thesis system._
