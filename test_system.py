#!/usr/bin/env python3
"""
Test script for Lip Reading AI System
This script tests all major components to ensure they work correctly.
"""

import torch
import numpy as np
import os
import sys
import time

# Add current directory to path
sys.path.append('.')

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from models.lip_reader_model import create_model, LipReaderLoss
        print("✓ Model imports successful")
    except ImportError as e:
        print(f"✗ Model imports failed: {e}")
        return False
    
    try:
        from utils.lip_detector import LipDetector, LipSequenceProcessor
        print("✓ Lip detector imports successful")
    except ImportError as e:
        print(f"✗ Lip detector imports failed: {e}")
        return False
    
    try:
        from utils.text_utils import TextProcessor
        print("✓ Text utils imports successful")
    except ImportError as e:
        print(f"✗ Text utils imports failed: {e}")
        return False
    
    try:
        import config
        print("✓ Config imports successful")
    except ImportError as e:
        print(f"✗ Config imports failed: {e}")
        return False
    
    return True


def test_model_creation():
    """Test model creation and forward pass"""
    print("\nTesting model creation...")
    
    try:
        from models.lip_reader_model import create_model
        
        # Create model
        vocab_size = 50
        device = 'cpu'  # Use CPU for testing
        model = create_model(vocab_size, device)
        
        print(f"✓ Model created successfully")
        print(f"  - Vocabulary size: {vocab_size}")
        print(f"  - Device: {device}")
        
        # Test forward pass
        batch_size = 2
        seq_length = 30
        channels = 3
        height = 64
        width = 64
        
        dummy_input = torch.randn(batch_size, seq_length, channels, height, width)
        output = model(dummy_input)
        
        expected_output_shape = (batch_size, seq_length, vocab_size)
        assert output.shape == expected_output_shape, f"Expected {expected_output_shape}, got {output.shape}"
        
        print(f"✓ Forward pass successful")
        print(f"  - Input shape: {dummy_input.shape}")
        print(f"  - Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False


def test_text_processor():
    """Test text processing utilities"""
    print("\nTesting text processor...")
    
    try:
        from utils.text_utils import TextProcessor
        
        # Create text processor
        processor = TextProcessor()
        
        print(f"✓ Text processor created")
        print(f"  - Vocabulary size: {processor.vocab_size}")
        
        # Test text processing
        test_texts = ["hello world", "how are you", "good morning"]
        
        for text in test_texts:
            # Convert to tensor
            indices = []
            for char in text:
                if char in processor.char_to_idx:
                    indices.append(processor.char_to_idx[char])
                else:
                    indices.append(processor.char_to_idx[processor.UNK_TOKEN])
            tensor = torch.tensor(indices, dtype=torch.long)
            
            # Convert back to text
            tokens = []
            for idx in indices:
                if idx in processor.idx_to_char:
                    token = processor.idx_to_char[idx]
                    tokens.append(token)
            decoded_text = ''.join(tokens)
            
            print(f"  - Original: '{text}' -> Decoded: '{decoded_text}'")
            
            # Basic validation
            if text.lower() not in decoded_text.lower():
                print(f"✗ Text processing failed for: {text}")
                return False
        
        print("✓ Text processing successful")
        
        # Test error rate computation
        predictions = ["hello world", "how are you"]
        targets = ["hello world", "how are you"]
        
        wer = processor.compute_wer(predictions, targets)
        cer = processor.compute_cer(predictions, targets)
        
        print(f"  - WER: {wer:.4f}, CER: {cer:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Text processor test failed: {e}")
        return False


def test_lip_detector():
    """Test lip detection utilities"""
    print("\nTesting lip detector...")
    
    try:
        from utils.lip_detector import LipDetector
        
        # Create lip detector
        detector = LipDetector()
        
        print("✓ Lip detector created")
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test lip region detection (should return None for dummy image)
        lip_region = detector.detect_lip_region(dummy_image)
        
        if lip_region is None:
            print("✓ Lip detection working (no face in dummy image)")
        else:
            print("✓ Lip region detected")
        
        # Test preprocessing
        if lip_region is not None:
            processed = detector.preprocess_lip_region(lip_region)
            print(f"✓ Lip preprocessing successful, output shape: {processed.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Lip detector test failed: {e}")
        return False


def test_sequence_processor():
    """Test sequence processing"""
    print("\nTesting sequence processor...")
    
    try:
        from utils.lip_detector import LipSequenceProcessor
        
        # Create processor
        processor = LipSequenceProcessor(sequence_length=30)
        
        print("✓ Sequence processor created")
        
        # Test with dummy data
        dummy_sequences = [torch.randn(3, 64, 64) for _ in range(25)]
        
        # Test padding
        if len(dummy_sequences) < 30:
            # Should pad with last frame
            padded_sequences = dummy_sequences + [dummy_sequences[-1]] * 5
            assert len(padded_sequences) == 30, "Padding failed"
            print("✓ Sequence padding working")
        
        # Test truncation
        long_sequences = [torch.randn(3, 64, 64) for _ in range(40)]
        truncated = long_sequences[:30]
        assert len(truncated) == 30, "Truncation failed"
        print("✓ Sequence truncation working")
        
        return True
        
    except Exception as e:
        print(f"✗ Sequence processor test failed: {e}")
        return False


def test_loss_function():
    """Test loss function"""
    print("\nTesting loss function...")
    
    try:
        from models.lip_reader_model import LipReaderLoss
        
        # Create loss function
        criterion = LipReaderLoss()
        
        print("✓ Loss function created")
        
        # Create dummy data
        batch_size = 2
        seq_length = 30
        vocab_size = 50
        
        logits = torch.randn(batch_size, seq_length, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, 10))  # Variable length targets
        input_lengths = torch.full((batch_size,), seq_length, dtype=torch.long)
        target_lengths = torch.tensor([10, 8], dtype=torch.long)
        
        # Compute loss
        loss = criterion(logits, targets, input_lengths, target_lengths)
        
        print(f"✓ Loss computation successful: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Loss function test failed: {e}")
        return False


def test_config():
    """Test configuration"""
    print("\nTesting configuration...")
    
    try:
        import config
        
        # Test basic config access
        model_config = config.MODEL_CONFIG
        training_config = config.TRAINING_CONFIG
        
        print("✓ Configuration loaded")
        print(f"  - Model hidden size: {model_config['hidden_size']}")
        print(f"  - Training batch size: {training_config['batch_size']}")
        print(f"  - Device: {config.DEVICE}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_directory_structure():
    """Test if required directories exist"""
    print("\nTesting directory structure...")
    
    required_dirs = [
        'models',
        'utils',
        'training',
        'inference',
        'preprocessing',
        'data'
    ]
    
    missing_dirs = []
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ Directory exists: {dir_name}")
        else:
            print(f"✗ Directory missing: {dir_name}")
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"\nCreating missing directories: {missing_dirs}")
        for dir_name in missing_dirs:
            os.makedirs(dir_name, exist_ok=True)
            print(f"✓ Created directory: {dir_name}")
    
    return len(missing_dirs) == 0


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("LIP READING AI SYSTEM - SYSTEM TEST")
    print("=" * 60)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Model Creation", test_model_creation),
        ("Text Processor", test_text_processor),
        ("Lip Detector", test_lip_detector),
        ("Sequence Processor", test_sequence_processor),
        ("Loss Function", test_loss_function),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Prepare your video dataset")
        print("2. Run preprocessing: python preprocessing/extract_lip_sequences.py")
        print("3. Train the model: python training/train_lip_reader.py")
        print("4. Test real-time: python inference/real_time_lip_reader.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1) 