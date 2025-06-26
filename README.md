# Lip Reading System

A modern deep learning pipeline for visual speech recognition (lip reading) using the [GRID Audio-Visual Speech Corpus](https://spandh.dcs.shef.ac.uk/gridcorpus/).

---

## Features
- **Direct training from GRID dataset**: No need for manual preprocessing or metadata files.
- **PyTorch-based**: Modular, extensible, and GPU-ready.
- **Supports multiple speakers**: Easily add more data for better generalization.
- **Modern model**: CNN + LSTM + Attention for robust sequence modeling.

---

## 1. Setup
#hiiiiiiii
### 1.1. Clone and Install Dependencies
```bash
# Clone the repo
https://github.com/yourusername/lip-reading.git
cd lip-reading

# Install Python dependencies
pip install -r requirements.txt
```

### 1.2. Download the GRID Corpus
- Request access and download from: [GRID Corpus](https://spandh.dcs.shef.ac.uk/gridcorpus/)
- Extract the dataset to: `data/GRID/`

Your folder should look like:
```
data/
  GRID/
    s1/
      video/
        *.mpg
      align/
        *.align
    s2/
      ...
```

---

## 2. Training

### 2.1. Basic Training Command
Train on a single speaker (e.g., `s1`):
```bash
python -m training.train_lip_reader --data_path data/GRID --epochs 10 --batch_size 4 --device cuda --train_speakers s1
```

- **Add more speakers**: `--train_speakers s1 s2 s3`
- **Validation**: By default, a split of the training data is used. To use a separate speaker for validation:
  ```bash
  python training/train_lip_reader.py --data_path data/GRID --train_speakers s1 --val_speakers s2
  ```
- **Best model** is saved as `best_lip_reader_model.pth`.

### 2.2. Arguments
- `--data_path`: Path to GRID data (default: `data/GRID`)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--device`: `cuda` or `cpu`
- `--train_speakers`: List of speakers for training (e.g., `s1 s2`)
- `--val_speakers`: List of speakers for validation (optional)

---

## 3. Data Loading
- The code uses `GRIDDdataset` to read videos and transcripts directly from the GRID folder structure.
- No need for metadata files or preprocessing scripts.
- Transcripts are automatically extracted from `.align` files.

---

## 4. Adding More Data
- Download and extract more speakers into `data/GRID/`.
- Add their names to `--train_speakers` or `--val_speakers` as needed.

---

## 5. Model
- Model architecture: CNN (spatial) + LSTM (temporal) + Attention.
- Vocabulary is automatically set using `TextProcessor`.
- Loss: CTC (Connectionist Temporal Classification).

---

## 6. Troubleshooting
- **CUDA out of memory**: Lower `--batch_size` or use `--device cpu`.
- **No data found**: Check your `data/GRID` structure and speaker names.
- **Training is slow**: Use a GPU if available, or reduce sequence length/batch size.

---

## 7. Inference & Evaluation
- (Coming soon) Scripts for real-time inference and evaluation.
- For now, use the saved model weights for your own inference pipeline.

---

## 8. Credits
- GRID Audio-Visual Speech Corpus: [http://spandh.dcs.shef.ac.uk/gridcorpus/](http://spandh.dcs.shef.ac.uk/gridcorpus/)
- Model and code: [Your Name/Team]

---

## 9. License
MIT License 