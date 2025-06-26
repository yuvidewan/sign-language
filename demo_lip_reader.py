#!/usr/bin/env python3
"""
Demo script for the Lip Reading AI System
This script demonstrates the basic functionality of the lip reading system.
"""

import torch
import numpy as np
import cv2
import os
import sys
import time

# Add parent directory to path
sys.path.append('.')

from models.lip_reader_model import create_model
from utils.lip_detector import LipDetector, LipSequenceProcessor
from utils.text_utils import TextProcessor


def demo_lip_detection():
    """
    Demo lip detection from webcam
    """
    print("=== Lip Detection Demo ===")
    print("This demo will show real-time lip detection from your webcam.")
    print("Press 'q' to quit.")
    
    lip_detector = LipDetector()
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Starting lip detection...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect lip region
        lip_region = lip_detector.detect_lip_region(frame)
        
        if lip_region is not None:
            # Draw bounding box around lip region
            x, y, w, h = cv2.boundingRect(lip_region)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Show lip region in separate window
            cv2.imshow('Lip Region', lip_region)
        
        # Draw lip landmarks
        lip_landmarks = lip_detector.extract_lip_landmarks(frame)
        if lip_landmarks:
            frame = lip_detector.draw_lip_landmarks(frame, lip_landmarks)
        
        # Show main frame
        cv2.imshow('Lip Detection Demo', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Lip detection demo completed.")


def demo_model_inference():
    """
    Demo model inference with dummy data
    """
    print("\n=== Model Inference Demo ===")
    print("This demo will show how the model processes lip sequences.")
    
    # Initialize components
    text_processor = TextProcessor()
    processor = LipSequenceProcessor()
    
    # Create dummy model
    vocab_size = text_processor.vocab_size
    model = create_model(vocab_size, device='cpu')
    model.eval()
    
    print(f"Created model with vocabulary size: {vocab_size}")
    
    # Create dummy lip sequence
    print("Creating dummy lip sequence...")
    dummy_sequence = torch.randn(1, 30, 3, 64, 64)  # (batch_size, seq_len, channels, height, width)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        logits = model(dummy_sequence)
        predictions = model.predict(dummy_sequence)
    
    print(f"Model output shape: {logits.shape}")
    print(f"Predictions shape: {predictions.shape}")
    
    # Decode predictions
    decoded_texts = text_processor.decode_ctc(logits)
    print(f"Decoded text: {decoded_texts[0] if decoded_texts else 'No prediction'}")
    
    print("Model inference demo completed.")


def demo_text_processing():
    """
    Demo text processing utilities
    """
    print("\n=== Text Processing Demo ===")
    print("This demo will show text processing capabilities.")
    
    # Initialize text processor
    text_processor = TextProcessor()
    
    # Sample texts
    sample_texts = [
        "hello world",
        "how are you",
        "good morning",
        "thank you very much"
    ]
    
    print("Sample texts:")
    for text in sample_texts:
        print(f"  - {text}")
    
    print("\nProcessing texts...")
    for text in sample_texts:
        # Convert to tensor
        tensor = text_processor.text_to_tensor(text)
        
        # Convert back to text
        decoded_text = text_processor.tensor_to_text(tensor)
        
        print(f"Original: '{text}'")
        print(f"Tensor shape: {tensor.shape}")
        print(f"Decoded: '{decoded_text}'")
        print()
    
    # Test error rate computation
    print("Testing error rate computation...")
    predictions = ["hello world", "how are you", "good morning"]
    targets = ["hello world", "how are you", "good evening"]
    
    wer = text_processor.compute_wer(predictions, targets)
    cer = text_processor.compute_cer(predictions, targets)
    
    print(f"Word Error Rate: {wer:.4f}")
    print(f"Character Error Rate: {cer:.4f}")
    
    print("Text processing demo completed.")


def demo_training_pipeline():
    """
    Demo training pipeline setup
    """
    print("\n=== Training Pipeline Demo ===")
    print("This demo will show how to set up the training pipeline.")
    
    # Create sample data structure
    sample_data_dir = "data/sample_dataset"
    os.makedirs(sample_data_dir, exist_ok=True)
    
    # Create sample metadata
    sample_metadata = {
        'dataset_info': {
            'total_videos': 5,
            'output_size': [64, 64],
            'sequence_length': 30,
            'processed_date': str(np.datetime64('now')),
            'note': 'Sample dataset for demonstration'
        },
        'samples': [
            {
                'video_path': 'sample_video_1.mp4',
                'text': 'hello world',
                'duration': 2.5,
                'sequence_length': 30
            },
            {
                'video_path': 'sample_video_2.mp4',
                'text': 'how are you',
                'duration': 3.0,
                'sequence_length': 30
            },
            {
                'video_path': 'sample_video_3.mp4',
                'text': 'good morning',
                'duration': 2.8,
                'sequence_length': 30
            },
            {
                'video_path': 'sample_video_4.mp4',
                'text': 'thank you',
                'duration': 2.2,
                'sequence_length': 30
            },
            {
                'video_path': 'sample_video_5.mp4',
                'text': 'please help',
                'duration': 2.7,
                'sequence_length': 30
            }
        ]
    }
    
    # Save metadata
    import json
    metadata_path = os.path.join(sample_data_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(sample_metadata, f, indent=2)
    
    print(f"Created sample dataset at: {sample_data_dir}")
    print(f"Metadata saved to: {metadata_path}")
    
    # Show training command
    print("\nTo train the model with this dataset, run:")
    print(f"python training/train_lip_reader.py --data_path {sample_data_dir} --epochs 10")
    
    print("Training pipeline demo completed.")


def main():
    """
    Main demo function
    """
    print("=" * 60)
    print("LIP READING AI SYSTEM - DEMO")
    print("=" * 60)
    print()
    print("This demo will showcase the main components of the lip reading system.")
    print("Choose which demo to run:")
    print("1. Lip Detection Demo (requires webcam)")
    print("2. Model Inference Demo")
    print("3. Text Processing Demo")
    print("4. Training Pipeline Demo")
    print("5. Run all demos")
    print("6. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-6): ").strip()
            
            if choice == '1':
                demo_lip_detection()
            elif choice == '2':
                demo_model_inference()
            elif choice == '3':
                demo_text_processing()
            elif choice == '4':
                demo_training_pipeline()
            elif choice == '5':
                print("\nRunning all demos...")
                demo_text_processing()
                demo_model_inference()
                demo_training_pipeline()
                print("\nNote: Skipping lip detection demo to avoid webcam issues.")
                print("Run option 1 separately if you want to test webcam functionality.")
            elif choice == '6':
                print("Exiting demo...")
                break
            else:
                print("Invalid choice. Please enter a number between 1 and 6.")
            
            print("\n" + "=" * 60)
            print("Demo completed. Choose another option or exit.")
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n\nDemo interrupted by user.")
            break
        except Exception as e:
            print(f"\nError during demo: {e}")
            print("Please try again or choose a different option.")
    
    print("\nThank you for trying the Lip Reading AI System demo!")
    print("To get started with your own data:")
    print("1. Prepare video files with clear lip movements")
    print("2. Create text transcripts for each video")
    print("3. Use the preprocessing scripts to prepare your dataset")
    print("4. Train the model using the training script")
    print("5. Use the real-time inference for live lip reading")


if __name__ == '__main__':
    main() 