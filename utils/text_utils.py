import torch
import re
from typing import List, Dict, Tuple
import json
import os
from typing import Optional


class TextProcessor:
    """
    Text processing utilities for lip reading
    """
    
    def __init__(self, vocab_path: Optional[str] = None, min_freq: int = 2):
        self.min_freq = min_freq
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'
        self.BLANK_TOKEN = '<BLANK>'  # For CTC
        
        if vocab_path and os.path.exists(vocab_path):
            self.load_vocabulary(vocab_path)
        else:
            # Initialize with basic vocabulary
            self._init_basic_vocab()
    
    def _init_basic_vocab(self):
        """
        Initialize with basic English vocabulary
        """
        # Basic English alphabet and common characters
        chars = list('abcdefghijklmnopqrstuvwxyz ')
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.BLANK_TOKEN]
        
        # Create vocabulary
        all_tokens = special_tokens + chars
        
        for idx, token in enumerate(all_tokens):
            self.char_to_idx[token] = idx
            self.idx_to_char[idx] = token
        
        self.vocab_size = len(all_tokens)
    
    def build_vocabulary(self, texts: List[str], save_path: Optional[str] = None):
        """
        Build vocabulary from text corpus
        Args:
            texts: List of text strings
            save_path: Path to save vocabulary
        """
        # Count character frequencies
        char_freq = {}
        
        for text in texts:
            text = self.preprocess_text(text)
            for char in text:
                char_freq[char] = char_freq.get(char, 0) + 1
        
        # Filter by minimum frequency
        filtered_chars = [char for char, freq in char_freq.items() if freq >= self.min_freq]
        
        # Create vocabulary
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.BLANK_TOKEN]
        all_tokens = special_tokens + sorted(filtered_chars)
        
        for idx, token in enumerate(all_tokens):
            self.char_to_idx[token] = idx
            self.idx_to_char[idx] = token
        
        self.vocab_size = len(all_tokens)
        
        # Save vocabulary
        if save_path:
            self.save_vocabulary(save_path)
    
    def save_vocabulary(self, path: str):
        """
        Save vocabulary to file
        """
        vocab_data = {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'vocab_size': self.vocab_size,
            'special_tokens': {
                'PAD': self.PAD_TOKEN,
                'UNK': self.UNK_TOKEN,
                'SOS': self.SOS_TOKEN,
                'EOS': self.EOS_TOKEN,
                'BLANK': self.BLANK_TOKEN
            }
        }
        
        with open(path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
    
    def load_vocabulary(self, path: str):
        """
        Load vocabulary from file
        """
        with open(path, 'r') as f:
            vocab_data = json.load(f)
        
        self.char_to_idx = vocab_data['char_to_idx']
        self.idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
        self.vocab_size = vocab_data['vocab_size']
        
        # Load special tokens
        special_tokens = vocab_data['special_tokens']
        self.PAD_TOKEN = special_tokens['PAD']
        self.UNK_TOKEN = special_tokens['UNK']
        self.SOS_TOKEN = special_tokens['SOS']
        self.EOS_TOKEN = special_tokens['EOS']
        self.BLANK_TOKEN = special_tokens['BLANK']
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for lip reading
        Args:
            text: Input text
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters except spaces
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def text_to_tensor(self, text: str, add_special_tokens: bool = True) -> torch.Tensor:
        """
        Convert text to tensor
        Args:
            text: Input text
            add_special_tokens: Whether to add SOS/EOS tokens
        Returns:
            Tensor of character indices
        """
        text = self.preprocess_text(text)
        
        if add_special_tokens:
            text = f"{self.SOS_TOKEN}{text}{self.EOS_TOKEN}"
        
        # Convert characters to indices (character by character, not word by word)
        indices = []
        for char in text:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                indices.append(self.char_to_idx[self.UNK_TOKEN])
        
        return torch.tensor(indices, dtype=torch.long)
    
    def tensor_to_text(self, tensor: torch.Tensor, remove_special_tokens: bool = True) -> str:
        """
        Convert tensor back to text
        Args:
            tensor: Input tensor
            remove_special_tokens: Whether to remove special tokens
        Returns:
            Decoded text
        """
        indices = tensor.tolist()
        tokens = []
        
        for idx in indices:
            if idx < len(self.idx_to_char):
                token = self.idx_to_char[idx]
                tokens.append(token)
        
        text = ''.join(tokens)  # Join characters directly, not with spaces
        
        if remove_special_tokens:
            # Remove special tokens
            text = text.replace(self.SOS_TOKEN, '').replace(self.EOS_TOKEN, '')
            text = text.replace(self.PAD_TOKEN, '').replace(self.UNK_TOKEN, '')
            text = text.replace(self.BLANK_TOKEN, '')
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def decode_ctc(self, logits: torch.Tensor, input_lengths: Optional[torch.Tensor] = None) -> List[str]:
        """
        Decode CTC output to text
        Args:
            logits: Model output logits (batch_size, seq_len, vocab_size)
            input_lengths: Length of each sequence
        Returns:
            List of decoded texts
        """
        batch_size = logits.size(0)
        decoded_texts = []
        
        # Get predictions
        predictions = torch.argmax(logits, dim=-1)
        
        for i in range(batch_size):
            if input_lengths is not None:
                seq_len = input_lengths[i]
                pred_seq = predictions[i, :seq_len]
            else:
                pred_seq = predictions[i]
            
            # Remove consecutive duplicates (CTC decoding)
            decoded_seq = []
            prev_token = None
            
            for token_idx in pred_seq:
                token_idx = token_idx.item()
                if token_idx != prev_token and token_idx != self.char_to_idx[self.BLANK_TOKEN]:
                    decoded_seq.append(token_idx)
                prev_token = token_idx
            
            # Convert to text
            text = self.tensor_to_text(torch.tensor(decoded_seq))
            decoded_texts.append(text)
        
        return decoded_texts
    
    def compute_wer(self, predictions: List[str], targets: List[str]) -> float:
        """
        Compute Word Error Rate
        Args:
            predictions: List of predicted texts
            targets: List of target texts
        Returns:
            Word Error Rate
        """
        total_errors = 0
        total_words = 0
        
        for pred, target in zip(predictions, targets):
            pred_words = pred.split()
            target_words = target.split()
            
            # Compute Levenshtein distance
            distance = self._levenshtein_distance(pred_words, target_words)
            total_errors += distance
            total_words += len(target_words)
        
        return total_errors / total_words if total_words > 0 else 0
    
    def compute_cer(self, predictions: List[str], targets: List[str]) -> float:
        """
        Compute Character Error Rate
        Args:
            predictions: List of predicted texts
            targets: List of target texts
        Returns:
            Character Error Rate
        """
        total_errors = 0
        total_chars = 0
        
        for pred, target in zip(predictions, targets):
            pred_chars = list(pred.replace(' ', ''))
            target_chars = list(target.replace(' ', ''))
            
            # Compute Levenshtein distance
            distance = self._levenshtein_distance(pred_chars, target_chars)
            total_errors += distance
            total_chars += len(target_chars)
        
        return total_errors / total_chars if total_chars > 0 else 0
    
    def _levenshtein_distance(self, seq1: List, seq2: List) -> int:
        """
        Compute Levenshtein distance between two sequences
        """
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
        
        return dp[m][n]


def create_sample_vocabulary():
    """
    Create a sample vocabulary for demonstration
    """
    processor = TextProcessor()
    
    # Sample texts for vocabulary building
    sample_texts = [
        "hello world",
        "how are you",
        "good morning",
        "thank you",
        "please help",
        "what time is it",
        "nice to meet you",
        "have a good day",
        "see you later",
        "goodbye"
    ]
    
    processor.build_vocabulary(sample_texts, 'sample_vocab.json')
    return processor 