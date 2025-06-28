import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import argparse
from tqdm import tqdm
from utils.grid_dataset import GRIDDdataset
from utils.text_utils import TextProcessor
from models.lip_reader_model import LipReaderModel, LipReaderLoss
import gc
import psutil
import time


def print_memory_usage():
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")


def collate_fn(batch):
    sequences = torch.stack([item[0] for item in batch])
    transcripts = [item[1] for item in batch]
    text_tensors = [text_processor.text_to_tensor(t, add_special_tokens=False) for t in transcripts]
    text_lengths = [len(t) for t in text_tensors]
    # Pad text_tensors to max length in batch
    max_len = max(text_lengths)
    padded_texts = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, t in enumerate(text_tensors):
        padded_texts[i, :len(t)] = t
    return {
        'sequences': sequences,
        'transcripts': transcripts,
        'text_tensors': padded_texts,
        'text_lengths': text_lengths
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc='Train')):
        try:
            sequences = batch['sequences'].to(device, non_blocking=True)
            targets = batch['text_tensors'].to(device, non_blocking=True)
            input_lengths = torch.full((sequences.size(0),), sequences.size(1), dtype=torch.long).to(device)
            target_lengths = torch.tensor(batch['text_lengths'], dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(sequences)
            loss = criterion(logits, targets, input_lengths, target_lengths)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            # Memory cleanup
            del sequences, targets, input_lengths, target_lengths, logits, loss
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Print memory usage every 10 batches
            if batch_idx % 10 == 0:
                print_memory_usage()
                
        except RuntimeError as e:
            if "out of memory" in str(e) or "not enough memory" in str(e):
                print(f"OOM error at batch {batch_idx}. Skipping batch and continuing...")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                continue
            else:
                raise e
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Val'):
            try:
                sequences = batch['sequences'].to(device, non_blocking=True)
                targets = batch['text_tensors'].to(device, non_blocking=True)
                input_lengths = torch.full((sequences.size(0),), sequences.size(1), dtype=torch.long).to(device)
                target_lengths = torch.tensor(batch['text_lengths'], dtype=torch.long).to(device)
                
                logits = model(sequences)
                loss = criterion(logits, targets, input_lengths, target_lengths)
                total_loss += loss.item()
                
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Lip Reader Model on GRID (Memory Optimized)')
    parser.add_argument('--data_path', type=str, default='data/GRID', help='Path to GRID data')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)  # Start with batch size 1
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--train_speakers', nargs='+', default=['s1'], help='Speakers for training')
    parser.add_argument('--val_speakers', nargs='+', default=None, help='Speakers for validation')
    parser.add_argument('--frames_per_clip', type=int, default=30)  # Reduced frames
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    # Set memory-efficient settings
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print_memory_usage()

    global text_processor
    text_processor = TextProcessor()
    num_classes = text_processor.vocab_size

    # Datasets with reduced frames per clip
    train_dataset = GRIDDdataset(root_dir=args.data_path, speakers=args.train_speakers, frames_per_clip=args.frames_per_clip)
    if args.val_speakers:
        val_dataset = GRIDDdataset(root_dir=args.data_path, speakers=args.val_speakers, frames_per_clip=args.frames_per_clip)
    else:
        val_size = max(1, int(0.1 * len(train_dataset)))
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
        pin_memory=False,
        persistent_workers=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False
    )

    # Model with gradient checkpointing
    model = LipReaderModel(num_classes)
    model = model.to(device)
    model.gradient_checkpointing_enable()
    
    criterion = LipReaderLoss(blank=text_processor.char_to_idx[text_processor.BLANK_TOKEN])
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        print_memory_usage()
        
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        epoch_time = time.time() - start_time
        
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.1f}s')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_lip_reader_model.pth')
            print('Saved best model.')
        
        # Clear memory after each epoch
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        print_memory_usage()
    
    print('Training complete!') 