import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append('..')

from models.lip_reader_model import create_model, LipReaderLoss
from utils.lip_detector import LipSequenceProcessor
from utils.text_utils import TextProcessor


class LipReadingDataset(Dataset):
    """
    Dataset for lip reading training
    """
    
    def __init__(self, data_dir, processor, text_processor, max_sequence_length=30):
        self.data_dir = data_dir
        self.processor = processor
        self.text_processor = text_processor
        self.max_sequence_length = max_sequence_length
        
        # Load dataset metadata
        self.samples = self._load_dataset()
        
    def _load_dataset(self):
        """
        Load dataset metadata from JSON file
        """
        metadata_path = os.path.join(self.data_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            # Create sample metadata for demonstration
            self._create_sample_metadata()
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata['samples']
    
    def _create_sample_metadata(self):
        """
        Create sample metadata for demonstration
        """
        # This would normally be created during data preprocessing
        sample_metadata = {
            'samples': [
                {
                    'video_path': 'sample_video_1.mp4',
                    'text': 'hello world',
                    'duration': 2.5
                },
                {
                    'video_path': 'sample_video_2.mp4', 
                    'text': 'how are you',
                    'duration': 3.0
                }
            ]
        }
        
        os.makedirs(self.data_dir, exist_ok=True)
        with open(os.path.join(self.data_dir, 'metadata.json'), 'w') as f:
            json.dump(sample_metadata, f, indent=2)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load video sequence
        video_path = os.path.join(self.data_dir, sample['video_path'])
        if not os.path.exists(video_path):
            # Create dummy data for demonstration
            sequence = torch.randn(self.max_sequence_length, 3, 64, 64)
        else:
            sequence = self.processor.process_video_sequence(video_path)
            if sequence is None:
                sequence = torch.randn(self.max_sequence_length, 3, 64, 64)
        
        # Process text
        text = sample['text']
        text_tensor = self.text_processor.text_to_tensor(text)
        
        return {
            'sequence': sequence,
            'text': text,
            'text_tensor': text_tensor,
            'text_length': len(text_tensor)
        }


def collate_fn(batch):
    """
    Custom collate function for batching
    """
    sequences = torch.stack([item['sequence'] for item in batch])
    texts = [item['text'] for item in batch]
    text_tensors = [item['text_tensor'] for item in batch]
    text_lengths = [item['text_length'] for item in batch]
    
    return {
        'sequences': sequences,
        'texts': texts,
        'text_tensors': text_tensors,
        'text_lengths': text_lengths
    }


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer):
    """
    Train for one epoch
    """
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(progress_bar):
        sequences = batch['sequences'].to(device)
        text_tensors = batch['text_tensors']
        text_lengths = batch['text_lengths']
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(sequences)
        
        # Prepare targets for CTC loss
        batch_size = sequences.size(0)
        seq_length = logits.size(1)
        
        # Create target tensor
        max_target_length = max(text_lengths)
        targets = torch.zeros(batch_size, max_target_length, dtype=torch.long)
        
        for i, text_tensor in enumerate(text_tensors):
            targets[i, :len(text_tensor)] = text_tensor
        
        # Create input and target lengths
        input_lengths = torch.full((batch_size,), seq_length, dtype=torch.long)
        target_lengths = torch.tensor(text_lengths, dtype=torch.long)
        
        # Compute loss
        loss = criterion(logits, targets, input_lengths, target_lengths)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}'
        })
        
        # Log to tensorboard
        if batch_idx % 10 == 0:
            writer.add_scalar('Training/Loss', loss.item(), epoch * num_batches + batch_idx)
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device, epoch, writer):
    """
    Validate the model
    """
    model.eval()
    total_loss = 0
    num_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            sequences = batch['sequences'].to(device)
            text_tensors = batch['text_tensors']
            text_lengths = batch['text_lengths']
            texts = batch['texts']
            
            # Forward pass
            logits = model(sequences)
            
            # Prepare targets
            batch_size = sequences.size(0)
            seq_length = logits.size(1)
            max_target_length = max(text_lengths)
            
            targets = torch.zeros(batch_size, max_target_length, dtype=torch.long)
            for i, text_tensor in enumerate(text_tensors):
                targets[i, :len(text_tensor)] = text_tensor
            
            input_lengths = torch.full((batch_size,), seq_length, dtype=torch.long)
            target_lengths = torch.tensor(text_lengths, dtype=torch.long)
            
            # Compute loss
            loss = criterion(logits, targets, input_lengths, target_lengths)
            total_loss += loss.item()
            
            # Compute accuracy (simplified)
            predictions = model.predict(sequences)
            # This is a simplified accuracy calculation
            # In practice, you'd use CTC decoding and compute WER/CER
            
            total_samples += batch_size
    
    avg_loss = total_loss / len(dataloader)
    accuracy = num_correct / total_samples if total_samples > 0 else 0
    
    writer.add_scalar('Validation/Loss', avg_loss, epoch)
    writer.add_scalar('Validation/Accuracy', accuracy, epoch)
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train Lip Reading Model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--output_dir', type=str, default='../models', help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--sequence_length', type=int, default=30, help='Sequence length')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize components
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    processor = LipSequenceProcessor(sequence_length=args.sequence_length)
    text_processor = TextProcessor()
    
    # Create datasets
    train_dataset = LipReadingDataset(
        os.path.join(args.data_path, 'train'),
        processor,
        text_processor,
        args.sequence_length
    )
    
    val_dataset = LipReadingDataset(
        os.path.join(args.data_path, 'val'),
        processor,
        text_processor,
        args.sequence_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Create model
    vocab_size = text_processor.vocab_size
    model = create_model(vocab_size, device=str(device))
    
    # Loss and optimizer
    criterion = LipReaderLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)
        
        # Validate
        val_loss, val_accuracy = validate(model, val_loader, criterion, device, epoch, writer)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'vocab_size': vocab_size
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f'Saved best model with validation loss: {val_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'vocab_size': vocab_size
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch + 1}.pth'))
    
    writer.close()
    print('Training completed!')


if __name__ == '__main__':
    main() 