# Lip Reading AI System

A deep learning-based system that converts lip movements to text using computer vision and sequence modeling.

## Features

- **Real-time lip detection** using MediaPipe and OpenCV
- **Deep learning model** combining CNN for feature extraction and LSTM for sequence modeling
- **Training pipeline** for custom datasets
- **Real-time inference** from video input
- **Silent speech recognition** - converts lip movements to text without audio

## Project Structure

```
sign_language/
├── data/                   # Training data and datasets
├── models/                 # Model definitions and saved models
├── utils/                  # Utility functions
├── training/               # Training scripts
├── inference/              # Real-time inference
├── preprocessing/          # Data preprocessing tools
└── notebooks/              # Jupyter notebooks for experimentation
```

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd sign_language
```

### 2. Set Up Virtual Environment

**Option A: Using venv (Recommended)**
```bash
# Create virtual environment
python -m venv lip_reader_env

# Activate virtual environment
# On Windows:
lip_reader_env\Scripts\activate
# On macOS/Linux:
source lip_reader_env/bin/activate
```

**Option B: Using conda**
```bash
# Create conda environment
conda create -n lip_reader_env python=3.9
conda activate lip_reader_env
```

### 3. Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
# Run system tests
python test_system.py
```

## Usage

### Training
```bash
python training/train_lip_reader.py --data_path data/dataset --epochs 100
```

### Real-time Inference
```bash
python inference/real_time_lip_reader.py
```

### Preprocessing Data
```bash
python preprocessing/extract_lip_sequences.py --video_path data/videos --output_path data/processed
```

### Interactive Demo
```bash
python demo_lip_reader.py
```

## Model Architecture

- **CNN Encoder**: Extracts spatial features from lip regions
- **LSTM Decoder**: Processes temporal sequences
- **Attention Mechanism**: Focuses on relevant lip movements
- **CTC Loss**: Handles alignment between lip movements and text

## Dataset Requirements

- Videos with clear lip movements
- Corresponding text transcripts
- Multiple speakers for robustness
- Various lighting conditions

## Performance

- Real-time processing at 30 FPS
- Accuracy varies by dataset quality
- Works best with clear, front-facing video

## Troubleshooting

### Common Issues

1. **MediaPipe Installation Issues**
   ```bash
   pip install mediapipe --upgrade
   ```

2. **CUDA/GPU Issues**
   ```bash
   # Install CPU-only PyTorch if GPU not available
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Webcam Access Issues**
   - Ensure webcam permissions are granted
   - Try different camera indices (0, 1, 2)

### Virtual Environment Management

**Deactivating the environment:**
```bash
deactivate  # For venv
conda deactivate  # For conda
```

**Removing the environment:**
```bash
# For venv
rm -rf lip_reader_env  # On macOS/Linux
rmdir /s lip_reader_env  # On Windows

# For conda
conda env remove -n lip_reader_env
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 