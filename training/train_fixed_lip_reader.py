#!/usr/bin/env python3
"""
Fixed Lip Reader Training Script
This script addresses the high loss issue with better training parameters and monitoring.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import time
import gc
import argparse
import os
from tqdm import tqdm

from models.lip_reader_model_lightweight import LightweightLipReaderModel, LipReaderLoss
from utils.grid_dataset import GRIDDdataset
from utils.text_utils import TextProcessor

def collate_fn(batch):
    """Custom collate function for variable length sequences"""
    sequences = []
    transcripts = []
    
    for frames, transcript in batch:
        sequences.append(frames)
        transcripts.append(transcript)
    
    # Pad sequences to max length
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    
    for seq in sequences:
        if seq.size(0) < max_len:
            # Pad with last frame
            padding = seq[-1].unsqueeze(0).repeat(max_len - seq.size(0), 1, 1, 1)
            padded_seq = torch.cat([seq, padding], dim=0)
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)
    
    # Stack sequences
    sequences = torch.stack(padded_sequences)
    
    return {
        'sequences': sequences,
        'transcripts': transcripts
    }

def train_epoch(model, dataloader, criterion, optimizer, text_processor, device, epoch):
    """Train for one epoch with better monitoring"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    # Progress bar
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1} Training')
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            sequences = batch['sequences'].to(device)
            transcripts = batch['transcripts']
            
            # Convert transcripts to tensors
            text_tensors = []
            target_lengths = []
            
            for transcript in transcripts:
                tensor = text_processor.text_to_tensor(transcript, add_special_tokens=False)
                text_tensors.append(tensor)
                target_lengths.append(len(tensor))
            
            # Pad text tensors
            max_text_len = max(len(t) for t in text_tensors)
            padded_texts = []
            
            for tensor in text_tensors:
                if len(tensor) < max_text_len:
                    padding = torch.full((max_text_len - len(tensor),), 
                                       text_processor.char_to_idx[text_processor.PAD_TOKEN], 
                                       dtype=torch.long)
                    padded_tensor = torch.cat([tensor, padding])
                else:
                    padded_tensor = tensor
                padded_texts.append(padded_tensor)
            
            targets = torch.stack(padded_texts).to(device)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long)
            input_lengths = torch.full((sequences.size(0),), sequences.size(1), dtype=torch.long)
            
            # Forward pass
            logits = model(sequences)
            loss = criterion(logits, targets, input_lengths, target_lengths)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
            
            # Memory cleanup
            del sequences, targets, input_lengths, target_lengths, logits, loss
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "out of memory" in str(e) or "not enough memory" in str(e):
                print(f"OOM error during training. Skipping batch...")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                continue
            else:
                raise e
    
    return total_loss / num_batches

def validate(model, dataloader, criterion, text_processor, device, epoch):
    """Validate with better monitoring"""
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    
    # Progress bar
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1} Validation')
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            try:
                sequences = batch['sequences'].to(device)
                transcripts = batch['transcripts']
                
                # Convert transcripts to tensors
                text_tensors = []
                target_lengths = []
                
                for transcript in transcripts:
                    tensor = text_processor.text_to_tensor(transcript, add_special_tokens=False)
                    text_tensors.append(tensor)
                    target_lengths.append(len(tensor))
                
                # Pad text tensors
                max_text_len = max(len(t) for t in text_tensors)
                padded_texts = []
                
                for tensor in text_tensors:
                    if len(tensor) < max_text_len:
                        padding = torch.full((max_text_len - len(tensor),), 
                                           text_processor.char_to_idx[text_processor.PAD_TOKEN], 
                                           dtype=torch.long)
                        padded_tensor = torch.cat([tensor, padding])
                    else:
                        padded_tensor = tensor
                    padded_texts.append(padded_tensor)
                
                targets = torch.stack(padded_texts).to(device)
                target_lengths = torch.tensor(target_lengths, dtype=torch.long)
                input_lengths = torch.full((sequences.size(0),), sequences.size(1), dtype=torch.long)
                
                logits = model(sequences)
                loss = criterion(logits, targets, input_lengths, target_lengths)
                total_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
                })
                
                # Memory cleanup
                del sequences, targets, input_lengths, target_lengths, logits, loss
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e) or "not enough memory" in str(e):
                    print(f"OOM error during validation. Skipping batch...")
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
    
    return total_loss / num_batches

def test_model_predictions(model, dataloader, text_processor, device, num_samples=3):
    """Test model predictions to verify learning"""
    model.eval()
    print("\n=== Testing Model Predictions ===")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
                
            sequences = batch['sequences'].to(device)
            transcripts = batch['transcripts']
            
            logits = model(sequences)
            predictions = torch.argmax(logits, dim=-1)
            
            print(f"\nSample {i+1}:")
            print(f"  Target: '{transcripts[0]}'")
            
            # Decode predictions
            decoded = text_processor.tensor_to_text(predictions[0])
            print(f"  Predicted: '{decoded}'")
            
            # Check prediction diversity
            unique_preds = torch.unique(predictions[0])
            print(f"  Unique predictions: {unique_preds.tolist()}")
            print(f"  Prediction diversity: {len(unique_preds)} unique tokens")
            
            # Check if predictions are reasonable
            if len(unique_preds) < 3:
                print(f"  ⚠️  WARNING: Low prediction diversity!")
            elif len(decoded.strip()) == 0:
                print(f"  ⚠️  WARNING: Empty prediction!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fixed Lip Reader Training')
    parser.add_argument('--data_path', type=str, default='data/GRID', help='Path to GRID data')
    parser.add_argument('--epochs', type=int, default=30)  # More epochs
    parser.add_argument('--batch_size', type=int, default=1)  # Smaller batch size
    parser.add_argument('--learning_rate', type=float, default=1e-3)  # Higher learning rate
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--train_speakers', nargs='+', default=['s1'], help='Speakers for training')
    parser.add_argument('--val_speakers', nargs='+', default=None, help='Speakers for validation')
    parser.add_argument('--frames_per_clip', type=int, default=40)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Initialize text processor
    text_processor = TextProcessor()
    num_classes = text_processor.vocab_size
    print(f'Vocabulary size: {num_classes}')
    print(f'Vocabulary: {list(text_processor.char_to_idx.keys())}')

    # Datasets
    train_dataset = GRIDDdataset(root_dir=args.data_path, speakers=args.train_speakers, frames_per_clip=args.frames_per_clip)
    if args.val_speakers:
        val_dataset = GRIDDdataset(root_dir=args.data_path, speakers=args.val_speakers, frames_per_clip=args.frames_per_clip)
    else:
        val_size = max(1, int(0.2 * len(train_dataset)))  # 20% for validation
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=False
    )

    # Model with better initialization
    model = LightweightLipReaderModel(num_classes, hidden_size=256, dropout=0.2)
    model = model.to(device)
    
    # Initialize weights properly
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
    
    model.apply(init_weights)
    print("✅ Model weights initialized properly")
    
    criterion = LipReaderLoss(blank=text_processor.char_to_idx[text_processor.BLANK_TOKEN])
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=2)

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 5
    
    print(f"\nStarting training with {args.epochs} epochs...")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    
    for epoch in range(args.epochs):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch+1}/{args.epochs}')
        print(f'{"="*60}')
        
        start_time = time.time()
        
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, text_processor, device, epoch)
        
        # Validation
        val_loss = validate(model, val_loader, criterion, text_processor, device, epoch)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Time: {epoch_time:.1f}s')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Test predictions every 3 epochs
        if (epoch + 1) % 3 == 0:
            test_model_predictions(model, val_loader, text_processor, device)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_fixed_lip_reader_model.pth')
            print('✅ Saved best model!')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
        
        # Clear memory
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    print(f'\n{"="*60}')
    print('🎉 TRAINING COMPLETED! 🎉')
    print(f'{"="*60}')
    print(f'✅ Final Train Loss: {train_loss:.4f}')
    print(f'✅ Best Val Loss: {best_val_loss:.4f}')
    print(f'✅ Model saved as: best_fixed_lip_reader_model.pth')
    print(f'{"="*60}')
    
    # Final prediction test
    print('\n=== Final Model Test ===')
    test_model_predictions(model, val_loader, text_processor, device, num_samples=5) 