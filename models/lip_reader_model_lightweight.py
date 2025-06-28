import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CTCLoss
from torch.utils.checkpoint import checkpoint
import numpy as np


class LightweightLipReaderModel(nn.Module):
    """
    Lightweight Lip Reading Model with reduced memory footprint
    """
    
    def __init__(self, num_classes, input_channels=3, hidden_size=128, num_layers=1, dropout=0.3):
        super(LightweightLipReaderModel, self).__init__()
        
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        # Reduced CNN Encoder
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        
        # We'll calculate the CNN output size dynamically
        self.cnn_output_size = None
        
        # Create a placeholder fc1 layer (will be replaced after first forward pass)
        self.fc1 = nn.Linear(1, hidden_size)  # Placeholder
        
        # Single layer LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False  # Use unidirectional to save memory
        )
        
        # Output layer
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def _calculate_cnn_output_size(self, x):
        """Calculate the output size of CNN layers"""
        with torch.no_grad():
            # Pass through CNN layers
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = self.dropout(x)
            
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = self.dropout(x)
            
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.pool(x)
            x = self.dropout(x)
            
            # Flatten to get the size
            return x.numel() // x.size(0)  # Divide by batch size to get per-sample size
        
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
        
        # Calculate CNN output size on first forward pass
        if self.cnn_output_size is None:
            self.cnn_output_size = self._calculate_cnn_output_size(x)
            # Replace the placeholder fc1 with the correct one
            self.fc1 = nn.Linear(self.cnn_output_size, self.hidden_size).to(x.device)
            print(f"CNN output size calculated: {self.cnn_output_size}")
        
        # CNN feature extraction with gradient checkpointing
        if self.training and hasattr(self, 'gradient_checkpointing_enabled') and self.gradient_checkpointing_enabled:
            x = checkpoint(self._cnn_forward, x)
        else:
            x = self._cnn_forward(x)
        
        # Ensure x is a tensor (should always be the case)
        if x is None:
            raise RuntimeError("CNN forward pass returned None")
        
        # Flatten and reshape back to sequence
        x = x.view(batch_size, seq_len, -1)
        
        # Linear projection
        x = F.relu(self.fc1(x))
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Output projection
        output = self.fc2(lstm_out)
        
        return output
    
    def _cnn_forward(self, x):
        """CNN forward pass for gradient checkpointing"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)
        
        return x
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.gradient_checkpointing_enabled = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        self.gradient_checkpointing_enabled = False
    
    def predict(self, x):
        """
        Simple prediction
        Args:
            x: Input tensor
        Returns:
            Predicted sequence
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
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


def create_lightweight_model(vocab_size, device='cuda'):
    """
    Factory function to create and initialize the lightweight model
    """
    model = LightweightLipReaderModel(
        num_classes=vocab_size,
        input_channels=3,
        hidden_size=128,
        num_layers=1,
        dropout=0.3
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