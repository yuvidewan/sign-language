"""
Configuration file for Lip Reading AI System
"""

import torch

# Model Configuration
MODEL_CONFIG = {
    'input_channels': 3,
    'hidden_size': 256,
    'num_layers': 2,
    'dropout': 0.5,
    'sequence_length': 30,
    'target_size': (64, 64),  # (height, width)
    'attention_heads': 8
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 8,
    'learning_rate': 0.001,
    'epochs': 100,
    'patience': 10,
    'min_delta': 0.001,
    'gradient_clip': 1.0,
    'weight_decay': 1e-5
}

# Data Configuration
DATA_CONFIG = {
    'train_split': 0.8,
    'val_split': 0.1,
    'test_split': 0.1,
    'min_sequence_length': 10,
    'max_sequence_length': 50,
    'sample_rate': 30,  # FPS
    'augmentation': True
}

# Preprocessing Configuration
PREPROCESSING_CONFIG = {
    'lip_padding': 20,
    'normalize': True,
    'resize_method': 'bilinear',
    'color_space': 'RGB'
}

# Inference Configuration
INFERENCE_CONFIG = {
    'confidence_threshold': 0.5,
    'beam_size': 5,
    'max_length': 50,
    'temperature': 1.0
}

# Device Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
PATHS = {
    'data_dir': 'data',
    'models_dir': 'models',
    'logs_dir': 'logs',
    'checkpoints_dir': 'checkpoints',
    'outputs_dir': 'outputs'
}

# Vocabulary Configuration
VOCAB_CONFIG = {
    'min_freq': 2,
    'max_vocab_size': 10000,
    'special_tokens': ['<PAD>', '<UNK>', '<SOS>', '<EOS>', '<BLANK>']
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'lip_reader.log'
}

# Validation Configuration
VALIDATION_CONFIG = {
    'metrics': ['loss', 'accuracy', 'wer', 'cer'],
    'save_best': True,
    'early_stopping': True
}

# Augmentation Configuration
AUGMENTATION_CONFIG = {
    'horizontal_flip': 0.5,
    'rotation': 10,  # degrees
    'brightness': 0.2,
    'contrast': 0.2,
    'noise': 0.05
}

# Model Checkpoints
CHECKPOINT_CONFIG = {
    'save_frequency': 10,  # Save every N epochs
    'keep_last': 5,  # Keep last N checkpoints
    'save_optimizer': True,
    'save_scheduler': True
}

# Real-time Configuration
REALTIME_CONFIG = {
    'buffer_size': 30,
    'prediction_interval': 2.0,  # seconds
    'smooth_predictions': True,
    'display_confidence': True
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'num_workers': 4,
    'pin_memory': True,
    'prefetch_factor': 2,
    'persistent_workers': True
}

# All configurations combined
CONFIG = {
    'model': MODEL_CONFIG,
    'training': TRAINING_CONFIG,
    'data': DATA_CONFIG,
    'preprocessing': PREPROCESSING_CONFIG,
    'inference': INFERENCE_CONFIG,
    'device': DEVICE,
    'paths': PATHS,
    'vocab': VOCAB_CONFIG,
    'logging': LOGGING_CONFIG,
    'validation': VALIDATION_CONFIG,
    'augmentation': AUGMENTATION_CONFIG,
    'checkpoint': CHECKPOINT_CONFIG,
    'realtime': REALTIME_CONFIG,
    'performance': PERFORMANCE_CONFIG
} 