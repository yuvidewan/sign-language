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
    for batch in tqdm(dataloader, desc='Train'):
        sequences = batch['sequences'].to(device)
        targets = batch['text_tensors'].to(device)
        input_lengths = torch.full((sequences.size(0),), sequences.size(1), dtype=torch.long).to(device)
        target_lengths = torch.tensor(batch['text_lengths'], dtype=torch.long).to(device)
        optimizer.zero_grad()
        logits = model(sequences)
        loss = criterion(logits, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Val'):
            sequences = batch['sequences'].to(device)
            targets = batch['text_tensors'].to(device)
            input_lengths = torch.full((sequences.size(0),), sequences.size(1), dtype=torch.long).to(device)
            target_lengths = torch.tensor(batch['text_lengths'], dtype=torch.long).to(device)
            logits = model(sequences)
            loss = criterion(logits, targets, input_lengths, target_lengths)
            total_loss += loss.item()
    return total_loss / len(dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Lip Reader Model on GRID')
    parser.add_argument('--data_path', type=str, default='data/GRID', help='Path to GRID data')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--train_speakers', nargs='+', default=['s1'], help='Speakers for training')
    parser.add_argument('--val_speakers', nargs='+', default=None, help='Speakers for validation (default: split train)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    global text_processor
    text_processor = TextProcessor()
    num_classes = text_processor.vocab_size

    # Datasets
    train_dataset = GRIDDdataset(root_dir=args.data_path, speakers=args.train_speakers)
    if args.val_speakers:
        val_dataset = GRIDDdataset(root_dir=args.data_path, speakers=args.val_speakers)
    else:
        # Split train set for validation if no val speakers
        val_size = max(1, int(0.1 * len(train_dataset)))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Model
    model = LipReaderModel(num_classes)
    model = model.to(device)
    criterion = LipReaderLoss(blank=text_processor.char_to_idx[text_processor.BLANK_TOKEN])
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_lip_reader_model.pth')
            print('Saved best model.')
    print('Training complete!') 