import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import argparse
from tqdm import tqdm
from utils.grid_dataset import GRIDDdataset
from utils.text_utils import TextProcessor
from models.lip_reader_model_lightweight import LightweightLipReaderModel, LipReaderLoss
import gc
import time


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


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1} - Training')
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            sequences = batch['sequences'].to(device)
            targets = batch['text_tensors'].to(device)
            input_lengths = torch.full((sequences.size(0),), sequences.size(1), dtype=torch.long).to(device)
            target_lengths = torch.tensor(batch['text_lengths'], dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            
            logits = model(sequences)
            loss = criterion(logits, targets, input_lengths, target_lengths)
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
            
            # Force garbage collection every few batches
            if batch_idx % 10 == 0:
                gc.collect()
                
        except RuntimeError as e:
            if "out of memory" in str(e) or "not enough memory" in str(e):
                print(f"OOM error at batch {batch_idx}. Skipping batch and continuing...")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                continue
            else:
                raise e
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1} - Validation')
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            try:
                sequences = batch['sequences'].to(device)
                targets = batch['text_tensors'].to(device)
                input_lengths = torch.full((sequences.size(0),), sequences.size(1), dtype=torch.long).to(device)
                target_lengths = torch.tensor(batch['text_lengths'], dtype=torch.long).to(device)
                
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
            targets = batch['text_tensors'].to(device)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Improved Lip Reader Training')
    parser.add_argument('--data_path', type=str, default='data/GRID', help='Path to GRID data')
    parser.add_argument('--epochs', type=int, default=20)  # More epochs
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=5e-4)  # Lower learning rate
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--train_speakers', nargs='+', default=['s1'], help='Speakers for training')
    parser.add_argument('--val_speakers', nargs='+', default=None, help='Speakers for validation')
    parser.add_argument('--frames_per_clip', type=int, default=40)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    global text_processor
    text_processor = TextProcessor()
    num_classes = text_processor.vocab_size
    print(f'Vocabulary size: {num_classes}')

    # Datasets
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

    # Model
    model = LightweightLipReaderModel(num_classes)
    model = model.to(device)
    model.gradient_checkpointing_enable()
    
    criterion = LipReaderLoss(blank=text_processor.char_to_idx[text_processor.BLANK_TOKEN])
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    best_val_loss = float('inf')
    training_history = []
    
    print(f"\nStarting training with {args.epochs} epochs...")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    
    for epoch in range(args.epochs):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch+1}/{args.epochs}')
        print(f'{"="*60}')
        
        start_time = time.time()
        
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validation
        val_loss = validate(model, val_loader, criterion, device, epoch)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        
        # Record history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'time': epoch_time
        })
        
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Time: {epoch_time:.1f}s')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Test predictions every 5 epochs
        if (epoch + 1) % 5 == 0:
            test_model_predictions(model, val_loader, text_processor, device)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_improved_lip_reader_model.pth')
            print('✅ Saved best model!')
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'training_history': training_history
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pth')
            print(f'💾 Saved checkpoint for epoch {epoch+1}')
        
        # Clear memory
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    print(f'\n{"="*60}')
    print('🎉 TRAINING COMPLETED SUCCESSFULLY! 🎉')
    print(f'{"="*60}')
    print(f'✅ Final Train Loss: {train_loss:.4f}')
    print(f'✅ Best Val Loss: {best_val_loss:.4f}')
    print(f'✅ Model saved as: best_improved_lip_reader_model.pth')
    print(f'✅ Training completed on speakers: {args.train_speakers}')
    print(f'{"="*60}')
    print('Your improved lip reading model is ready to use!')
    print(f'{"="*60}')
    
    # Final prediction test
    print('\n=== Final Model Test ===')
 