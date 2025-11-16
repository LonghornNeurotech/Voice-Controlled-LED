# Voice-Controlled LED - Raspberry Pi 4B Edge AI Project

A machine learning project that enables voice-activated LED control using a Raspberry Pi 4B. The system detects the keywords "Pi on" and "Pi off" using a quantized PyTorch model running directly on the Raspberry Pi, demonstrating edge AI capabilities for keyword detection.

## Table of Contents

- [Overview](#overview)
- [Materials](#materials)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Hardware Configuration](#hardware-configuration)
- [Data Collection](#data-collection)
- [Model Training](#model-training)
- [Real-Time Inference](#real-time-inference)
- [Requirements](#requirements)
- [Usage](#usage)
- [Deliverables](#deliverables)
- [Contributing](#contributing)

## Overview

This project implements a voice-controlled LED system that:
- Acquires audio signals using both a USB microphone (SunFounder USB 2.0 Mini Microphone) and an INMP441 MEMS I2S microphone module
- Trains a quantized and pruned machine learning model to detect "Pi on" and "Pi off" commands
- Runs real-time inference on the Raspberry Pi 4B to control LEDs based on voice commands
- Demonstrates edge AI capabilities with quantization-aware training and pruning for efficient deployment

The system classifies audio into three categories:
1. **"Pi on"** - Turns LED on
2. **"Pi off"** - Turns LED off  
3. **Background/Neutral** - No action (handles unrelated speech and noise)

## Materials

### Required Components
- **Raspberry Pi 4B** 
- **SunFounder USB 2.0 Mini Microphone** ([Amazon Link](https://www.amazon.com/SunFounder-Microphone-Raspberry-Recognition-Software/dp/B01KLRBHGM)) - Plug-and-play USB microphone for data collection
- **INMP441 MEMS I2S Microphone Module** ([Amazon Link](https://www.amazon.com/AITRIPAITRIP-AITRIP-Omnidirectional-Microphone-Interface/dp/B0972XP1YS)) - I2S interface microphone for real-time inference
- **2 LEDs** (one for USB mic recorder, one for INMP441 inference)
- **2 Resistors** (470Ω recommended, or appropriate for your LEDs)
- **Jumper wires**
- **Solderless breadboard**
- **Ethernet cable**

### Software Requirements
- Python 3.x
- PyTorch (with quantization support)
- Required Python packages (see `requirements.txt`)

## Project Structure

```
Voice-Controlled-LED/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── voice-recorder.py         # Data collection script (USB mic)
├── led_control.py            # GPIO LED toggle script (for USB mic recorder)
├── led_control_inmp441.py    # GPIO LED control for INMP441 inference
├── inference.py              # Real-time inference script
├── microphone_driver.py      # INMP441 I2S driver
├── validate_data.py          # Data validation script
├── model/
│   ├── architecture.py         # PyTorch model definition
│   └── weights/              # Trained model weights (.pt/.pth files) - created during training
├── data/                     # Audio data directory - create and add your recordings
│   ├── pi_on/                # "Pi on" audio samples
│   ├── pi_off/               # "Pi off" audio samples
│   └── background/           # Background noise samples
├── preprocessing/
│   ├── __init__.py           # Module initialization
│   └── preprocessing.py      # Master preprocessing script (team collaborates here)
├── training/
│   ├── __init__.py           # Module initialization
│   └── training.py           # Master training script (team collaborates here)
└── colab_training.ipynb      # Colab training notebook (optional)
```

## Setup Instructions

### 1. Raspberry Pi 4B Configuration

#### a) Initial Setup
- Install Raspberry Pi OS on the board if not already installed
- Ensure you have necessary cables/wires to connect to the board
- Set up SSH or direct access to the Pi

#### b) Python and PyTorch Installation
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip -y

# Install PyTorch for Raspberry Pi (ARM architecture)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip3 install -r requirements.txt
```

### 2. Hardware Connections

#### LED Circuits

**LED 1 - USB Microphone Recorder** (controlled by `voice-recorder.py`):
- Connect LED cathode (short leg) to ground via resistor (470Ω)
- Connect LED anode (long leg) to GPIO pin (e.g., GPIO 18)
- **GPIO Pin**: Documented in `led_control.py` (default: GPIO 18)
- **Resistor Value**: 470Ω (or appropriate for your LED)

**LED 2 - INMP441 Inference** (controlled by `inference.py`):
- Connect LED cathode (short leg) to ground via resistor (470Ω)
- Connect LED anode (long leg) to GPIO pin (e.g., GPIO 21)
- **GPIO Pin**: Documented in `led_control_inmp441.py` (default: GPIO 21)
- **Resistor Value**: 470Ω (or appropriate for your LED)

**LED Specifications**: Document model number and safe operating voltage range for both LEDs

#### INMP441 Microphone Connections
The INMP441 uses I2S (Inter-IC Sound) protocol with three data lines plus power:

| INMP441 Pin | Raspberry Pi Pin | Description |
|------------|-----------------|-------------|
| VDD        | 3.3V            | Power       |
| GND        | GND             | Ground      |
| WS (LRCL)  | GPIO 19         | Word Select |
| SCK (BCLK) | GPIO 18         | Bit Clock   |
| SD         | GPIO 20         | Serial Data |

**I2S Protocol Reference**: [Introduction to I2S Interface](https://www.allaboutcircuits.com/technical-articles/introduction-to-the-i2s-interface/)

### 3. Microphone Setup

#### USB Microphone (SunFounder)
- **Plug-and-play**: Simply plug the USB microphone into any USB port on the Raspberry Pi
- No driver installation required - Raspberry Pi OS will automatically detect it
- **Note**: Best used in a port that's not near the computer's cooling fan
- Use `sounddevice` or `pyaudio` to access the USB microphone in Python

#### INMP441 I2S Microphone
The INMP441 driver can be implemented in Python or integrated from C code using `ctypes`. Enable I2S interface on Raspberry Pi:

```bash
# Enable I2S in raspi-config
sudo raspi-config
# Navigate to: Interface Options → I2S → Enable
```

## Data Collection

### Protocol
- **Sampling Frequency**: 16 kHz (standard for speech)
- **Sample Duration**: 1.5-2 seconds per recording
- **Number of Samples**: Minimum 25 per class
- **Classes**: 
  - "Pi on"
  - "Pi off"
  - Background/neutral (unrelated speech, noise, silence)

### Usage
```bash
python3 voice-recorder.py
```

**Controls**:
- Press `1` → Record "Pi on"
- Press `2` → Record "Pi off"
- Press `3` → Record background noise
- Press `q` → Quit

**Tips for Data Collection**:
- Vary speaking speed, tone, and volume
- Record in different environments
- Include multiple speakers if possible
- Capture various background conditions
- Store samples in organized folder structure: `data/{class_name}/` (or `voice_data/{class_name}/` if using default voice-recorder.py)

### Data Validation
After collection, verify audio files are not corrupted:
```bash
# Check audio files
python3 validate_data.py
```

## Model Training

### Collaboration Workflow
All team members collaborate on a single master file using Git branches and Pull Requests:
- **Preprocessing**: `preprocessing/preprocessing.py` - Master preprocessing script
- **Training**: `training/training.py` - Master training script

**Workflow**:
1. Create a feature branch: `git checkout -b feature-name`
2. Edit the master file(s) with your contributions
3. Commit and push: `git push origin feature-name`
4. Open a Pull Request for team review
5. Merge after approval

All functionality is consolidated in the master files:
- `preprocessing/preprocessing.py` - Contains all preprocessing classes and utilities
- `training/training.py` - Contains all training classes and utilities

### Preprocessing
The master preprocessing script (`preprocessing/preprocessing.py`) contains the `VoicePreprocessor` and `VoiceDataset` classes. Team members should contribute preprocessing steps such as:
- Normalization
- Feature extraction (MFCC, spectrograms, mel-spectrograms, etc.)
- Data augmentation (time shifting, pitch shifting, noise injection, etc.)
- Windowing and framing
- Filtering and denoising

### Training Process
The master training script (`training/training.py`) provides a simple training pipeline:
1. **Train/Validation Split**: Automatically splits data into training and validation sets
2. **Training Loop**: Standard PyTorch training with loss and accuracy tracking
3. **Model Saving**: Saves the best model based on validation accuracy
4. **Model Architecture**: Defined in `model/architecture.py`

```bash
# Run the master training script
python3 training/training.py --data_dir ./data --epochs 50 --batch_size 32 --lr 0.001
```

**Training Features**:
- Automatic train/validation split (default: 80/20)
- Best model checkpointing based on validation accuracy
- Progress tracking with loss and accuracy metrics
- GPU support (automatically uses CUDA if available)

### Model Interpretability
- Analyze feature importance
- Visualize neuron conductances
- Generate attribution maps highlighting relevant signal regions
- Identify frequencies most important for predictions

### Model Visualization
- Use `torchviz` or similar library to visualize architecture
- Alternative: Use `torchsummary` for model summary

### Model Export
- Save best model as `.pt` or `.pth` file
- **Bonus**: Export as `.pte` file using Executorch for optimized deployment

## ⚡ Real-Time Inference

### Deployment to Raspberry Pi
1. Transfer trained model weights to Pi
2. Ensure all dependencies are installed
3. Run inference script:

```bash
python3 inference.py --model_path ./model/weights/best_model.pt
```

### Inference Pipeline
1. **Audio Acquisition**: Continuous buffering from INMP441 microphone
2. **Preprocessing**: Apply same preprocessing as training (using `VoicePreprocessor` from `preprocessing/preprocessing.py`)
3. **Model Inference**: Run quantized and pruned model on audio buffer
4. **Post-processing**: Smooth predictions to reduce flickering
5. **LED Control**: Update LED state based on prediction (controls LED 2 via `led_control_inmp441.py`)
6. **Loop**: Infinite loop for continuous operation

### Post-Processing
Implement smoothing/filtering to prevent LED flickering:
- Moving average of predictions
- Confidence thresholding
- State change debouncing

## 📦 Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

**Key Dependencies**:
- PyTorch (with quantization support)
- NumPy
- SoundDevice / PyAudio (for audio I/O)
- SciPy (for signal processing)
- RPi.GPIO (for Raspberry Pi GPIO control)
- Other ML/audio processing libraries

See `requirements.txt` for complete list with versions.

## 💻 Usage

### 1. LED Toggle Test
```bash
# Test LED 1 (USB mic recorder)
python3 led_control.py

# Test LED 2 (INMP441 inference)
python3 led_control_inmp441.py
```

### 2. Microphone Test
```bash
# Test USB microphone
python3 -c "import sounddevice as sd; print(sd.query_devices())"
# Select USB microphone device and record test audio

# Test INMP441 I2S microphone
python3 microphone_driver.py --test
# Record a few seconds, transfer to computer, and play back
```

### 3. Data Collection
```bash
python3 voice-recorder.py
```

### 4. Model Training (on development machine)
```bash
# Run the master training script
python3 training/training.py --data_dir ./data --epochs 50 --batch_size 32 --lr 0.001
```

### 5. Real-Time Inference (on Raspberry Pi)
```bash
python3 inference.py
```

### 1. Working System
- ✅ Keyword detection model running on Raspberry Pi 4B
- ✅ LED responds to "Pi on" and "Pi off" commands
- ✅ Handles background noise without false triggers

### 2. GitHub Repository
- ✅ Well-organized code structure
- ✅ Microphone driver
- ✅ Data collection scripts
- ✅ Model architecture and weights
- ✅ LED control scripts
- ✅ Real-time inference pipeline
- ✅ `requirements.txt` with versions


### 3. Demo Video
Short video demonstrating:
- Saying "Pi on" → LED turns on
- Saying "Pi off" → LED turns off
- Background noise/conversation → LED remains unchanged


## Related Projects

- **EEG Motor Imagery Classification**: [Dataset](https://www.kaggle.com/datasets/aymanmostafa11/eeg-motor-imagery-bciciv-2a)
- **EMG Gesture Recognition**: [Dataset](https://physionet.org/content/grabmyo/1.1.1/#files-panel)
