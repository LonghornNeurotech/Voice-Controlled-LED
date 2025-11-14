"""
PyTorch model architecture for voice-controlled LED project.
Team members should collaborate on this file to design and improve the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VoiceClassifier(nn.Module):
    """
    Neural network model for classifying voice commands.
    
    This is a template architecture. Team members should collaborate to:
    - Adjust architecture (layers, sizes, activation functions)
    - Add regularization (dropout, batch norm)
    - Optimize for edge deployment (quantization-friendly)
    
    Args:
        input_size: Size of input features (e.g., flattened MFCC or spectrogram)
        num_classes: Number of output classes (default: 3 for pi_on, pi_off, background)
    """
    
    def __init__(self, input_size: int = 128, num_classes: int = 3):
        super(VoiceClassifier, self).__init__()
        
        # TODO: Team members should collaborate to design the optimal architecture
        # Example architecture - adjust based on your feature extraction
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.2)
        
        # Output layer
        self.fc4 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Flatten input if needed (in case of 2D features)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Forward through layers
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        
        # Output logits
        x = self.fc4(x)
        
        return x