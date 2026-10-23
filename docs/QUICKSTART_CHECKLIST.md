# Quick-Start Checklist

Follow these steps in order.

## 1. Environment

- [ ] Create & activate virtual environment
- [ ] `pip install -r requirements.txt`
- [ ] Install system PortAudio (see TOOLS_AND_SETUP.md)
- [ ] Verify imports (tensorflow, mediapipe, librosa, cv2, pyaudio)

## 2. Data

- [ ] Download FER-2013 (Kaggle or manual) → place `fer2013.csv` in `data/raw/`
- [ ] Run `python scripts/prepare_fer2013.py`
- [ ] Download RAVDESS speech zip → extract so `data/raw/ravdess/Actor_*` exists
- [ ] Run `python scripts/prepare_ravdess.py`

## 3. Training

- [ ] `python scripts/train_video.py` (GPU recommended)
- [ ] `python scripts/train_audio.py`
- [ ] (Optional) `python scripts/evaluate.py` to see test metrics & confusion matrices

## 4. Real-time demo

- [ ] Make sure webcam and microphone work
- [ ] `python scripts/run_realtime.py`
- [ ] Press `q` to quit, `r` to reset fusion history

## 5. Interview / Portfolio

- [ ] Read `docs/ARCHITECTURE_AND_DESIGN.md`
- [ ] Be ready to explain late fusion, why 48×48, MediaPipe vs Haar, and the public-dataset choice
- [ ] Mention limitations honestly (acted data, cultural bias, etc.)
