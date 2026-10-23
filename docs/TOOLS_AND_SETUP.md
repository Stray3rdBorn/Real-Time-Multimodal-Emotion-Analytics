# Tools, Libraries & Setup Checklist

## Required Software

| Tool / Library                | Purpose                    | Install note                       |
| ----------------------------- | -------------------------- | ---------------------------------- |
| Python 3.10 – 3.12            | Runtime                    | –                                  |
| TensorFlow ≥ 2.15             | Deep learning              | `pip install tensorflow`           |
| Keras 3                       | High-level API             | comes with TF                      |
| OpenCV                        | Webcam & image ops         | `opencv-python`                    |
| MediaPipe                     | Face detection             | `mediapipe`                        |
| librosa + soundfile           | Audio loading & MFCC       | –                                  |
| PyAudio                       | Real-time microphone       | needs PortAudio system lib         |
| NumPy / Pandas / scikit-learn | Data handling              | –                                  |
| Matplotlib + Seaborn          | Plots & confusion matrices | –                                  |
| tqdm                          | Progress bars              | –                                  |
| Kaggle API (optional)         | Easy FER-2013 download     | `kaggle` + `~/.kaggle/kaggle.json` |

## System-level dependencies

**Ubuntu / Debian**

```bash
sudo apt update
sudo apt install -y portaudio19-dev python3-pyaudio ffmpeg libsndfile1
```

**macOS**

```bash
brew install portaudio ffmpeg
```

**Windows**

- Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- Or use Conda: `conda install -c conda-forge pyaudio`

## Python environment (recommended)

```bash
cd emotion_analytics
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset download links

1. **FER-2013**  
   https://www.kaggle.com/datasets/msambare/fer2013  
   (or use the Kaggle API as shown in the README)

2. **RAVDESS Speech**  
   https://zenodo.org/record/1188976  
   File: `Audio_Speech_Actors_01-24.zip`

## GPU (optional but strongly recommended for training)

- NVIDIA GPU + CUDA 12.x + cuDNN matching your TensorFlow version
- Or use Google Colab / Kaggle notebooks for the training scripts

## Verification checklist after setup

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import mediapipe; print('MediaPipe OK')"
python -c "import librosa; print('librosa OK')"
python -c "import cv2; print(cv2.__version__)"
python -c "import pyaudio; print('PyAudio OK')"
```
