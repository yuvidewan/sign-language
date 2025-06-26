import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CTCLoss
import numpy as np


class LipReaderModel(nn.Module):
    """
    Lip Reading Model combining CNN for spatial feature extraction
    and LSTM for temporal sequence modeling
    """
    
    def __init__(self, num_classes, input_channels=3, hidden_size=256, num_layers=2, dropout=0.5):
        super(LipReaderModel, self).__init__()
        
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        # CNN Encoder for spatial feature extraction
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        
        # Calculate the size after CNN layers (assuming 64x64 input)
        # After 4 pooling layers: 64 -> 32 -> 16 -> 8 -> 4
        # So feature map size is 4x4 with 512 channels
        self.cnn_output_size = 4 * 4 * 512
        
        # Linear layer to reduce dimensions before LSTM
        self.fc1 = nn.Linear(self.cnn_output_size, hidden_size)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Output layer
        self.fc2 = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=8, dropout=dropout)
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, sequence_length, channels, height, width)
        Returns:
            Output tensor of shape (batch_size, sequence_length, num_classes)
        """
        batch_size, seq_len, channels, height, width = x.size()
        
        # Reshape for CNN processing
        x = x.view(batch_size * seq_len, channels, height, width)
        
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Flatten and reshape back to sequence
        x = x.view(batch_size, seq_len, -1)
        
        # Linear projection
        x = F.relu(self.fc1(x))
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        lstm_out = lstm_out.transpose(0, 1)  # (seq_len, batch_size, hidden_size*2)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out.transpose(0, 1)  # (batch_size, seq_len, hidden_size*2)
        
        # Output projection
        output = self.fc2(attn_out)
        
        return output
    
    def predict(self, x, beam_size=5):
        """
        Beam search prediction
        Args:
            x: Input tensor
            beam_size: Number of beams for search
        Returns:
            Predicted sequence
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            
            # Simple greedy decoding (can be improved with beam search)
            predictions = torch.argmax(probs, dim=-1)
            
        return predictions


class LipReaderLoss(nn.Module):
    """
    Custom loss function for lip reading
    """
    
    def __init__(self, blank=0):
        super(LipReaderLoss, self).__init__()
        self.ctc_loss = CTCLoss(blank=blank, zero_infinity=True)
        
    def forward(self, logits, targets, input_lengths, target_lengths):
        """
        Compute CTC loss
        Args:
            logits: Model output (batch_size, seq_len, num_classes)
            targets: Target sequences
            input_lengths: Length of each input sequence
            target_lengths: Length of each target sequence
        """
        # Convert logits to log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Transpose for CTC: (seq_len, batch_size, num_classes)
        log_probs = log_probs.transpose(0, 1)
        
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)


def create_model(vocab_size, device='cuda'):
    """
    Factory function to create and initialize the model
    """
    model = LipReaderModel(
        num_classes=vocab_size,
        input_channels=3,
        hidden_size=256,
        num_layers=2,
        dropout=0.5
    )
    
    # Initialize weights
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    return model.to(device) 