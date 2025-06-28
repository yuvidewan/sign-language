#!/usr/bin/env python3
"""
Real-time Lip Reader using trained model
"""

import torch
import cv2
import numpy as np
import argparse
import time
import sys
import os

# Add parent directory to path
sys.path.append('.')

from models.lip_reader_model_lightweight import LightweightLipReaderModel
from utils.lip_detector import LipDetector
from utils.text_utils import TextProcessor


class RealTimeLipReader:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        
        # Initialize components
        self.text_processor = TextProcessor()
        self.lip_detector = LipDetector()
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = LightweightLipReaderModel(self.text_processor.vocab_size)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        print("Model loaded successfully!")
        
        # Initialize sequence buffer
        self.sequence_length = 40  # Match training
        self.lip_sequence = []
        self.current_text = ""
        self.frame_count = 0
        
    def preprocess_frame(self, frame):
        """Preprocess a single frame for the model"""
        try:
            # Detect lip region
            lip_region = self.lip_detector.detect_lip_region(frame)
            
            if lip_region is None:
                return None
            
            # Resize to 64x64 (match training)
            resized = cv2.resize(lip_region, (64, 64))
            
            # Convert BGR to RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            normalized = rgb.astype(np.float32) / 255.0
            
            # Convert to tensor (C, H, W)
            tensor = torch.from_numpy(normalized).permute(2, 0, 1)
            
            return tensor
            
        except Exception as e:
            print(f"Error preprocessing frame: {e}")
            return None
    
    def predict_text(self, lip_sequence):
        """Predict text from lip sequence"""
        if len(lip_sequence) < 10:  # Need minimum frames
            return ""
        
        try:
            # Pad or truncate to sequence_length
            if len(lip_sequence) > self.sequence_length:
                lip_sequence = lip_sequence[-self.sequence_length:]
            elif len(lip_sequence) < self.sequence_length:
                # Pad with last frame
                last_frame = lip_sequence[-1] if lip_sequence else torch.zeros(3, 64, 64)
                while len(lip_sequence) < self.sequence_length:
                    lip_sequence.append(last_frame)
            
            # Stack frames into batch
            sequence_tensor = torch.stack(lip_sequence).unsqueeze(0)  # (1, seq_len, C, H, W)
            sequence_tensor = sequence_tensor.to(self.device)
            
            # Run inference
            with torch.no_grad():
                logits = self.model(sequence_tensor)
                predictions = torch.argmax(logits, dim=-1)
            
            # Decode predictions
            predicted_text = self.text_processor.tensor_to_text(predictions[0])
            
            return predicted_text
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return ""
    
    def run(self):
        """Run real-time lip reading"""
        print("Starting real-time lip reader...")
        print("Press 'q' to quit, 'r' to reset sequence")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set webcam properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        last_prediction_time = time.time()
        prediction_interval = 2.0  # Predict every 2 seconds
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break
            
            self.frame_count += 1
            
            # Preprocess frame
            processed_frame = self.preprocess_frame(frame)
            
            if processed_frame is not None:
                # Add to sequence
                self.lip_sequence.append(processed_frame)
                
                # Keep only recent frames
                if len(self.lip_sequence) > self.sequence_length:
                    self.lip_sequence = self.lip_sequence[-self.sequence_length:]
                
                # Draw lip detection box
                height, width = frame.shape[:2]
                center_x, center_y = width // 2, height // 2
                lip_size = 120
                
                x1 = max(0, center_x - lip_size // 2)
                y1 = max(0, center_y - lip_size // 2)
                x2 = min(width, center_x + lip_size // 2)
                y2 = min(height, center_y + lip_size // 2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Lip Detected", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Make prediction periodically
                current_time = time.time()
                if current_time - last_prediction_time > prediction_interval and len(self.lip_sequence) >= 10:
                    predicted_text = self.predict_text(self.lip_sequence)
                    if predicted_text.strip():
                        self.current_text = predicted_text
                        print(f"Predicted: {predicted_text}")
                    last_prediction_time = current_time
            else:
                cv2.putText(frame, "No Lip Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display current text
            if self.current_text:
                # Split text into lines for display
                words = self.current_text.split()
                lines = []
                current_line = ""
                
                for word in words:
                    if len(current_line + " " + word) < 30:
                        current_line += " " + word if current_line else word
                    else:
                        lines.append(current_line)
                        current_line = word
                
                if current_line:
                    lines.append(current_line)
                
                # Display lines
                for i, line in enumerate(lines):
                    y_pos = height - 100 + i * 30
                    cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(frame, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
            
            # Display info
            cv2.putText(frame, f"Frames: {len(self.lip_sequence)}/{self.sequence_length}", (10, height-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('Real-time Lip Reader', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.lip_sequence = []
                self.current_text = ""
                print("Sequence reset")
        
        cap.release()
        cv2.destroyAllWindows()
        print("Real-time lip reader stopped.")


def main():
    parser = argparse.ArgumentParser(description='Real-time Lip Reader')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    # Create and run lip reader
    lip_reader = RealTimeLipReader(args.model_path, args.device)
    lip_reader.run()


if __name__ == "__main__":
    main() 