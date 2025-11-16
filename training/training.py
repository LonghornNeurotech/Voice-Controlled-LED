"""
Master training script for voice-controlled LED project.
All team members should collaborate on this single file using Git branches and Pull Requests.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.preprocessing import VoiceDataset, VoicePreprocessor


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': 100. * correct / total
    }


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': 100. * correct / total
    }


def main():
    parser = argparse.ArgumentParser(description='Train voice-controlled LED model')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing audio data')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    parser.add_argument('--save_path', type=str, default='model/weights/best_model.pt',
                        help='Path to save best model')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize preprocessing and dataset
    preprocessor = VoicePreprocessor(sample_rate=16000)
    dataset = VoiceDataset(args.data_dir, preprocessor)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Split dataset into train and validation
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train samples: {train_size}, Validation samples: {val_size}")
    
    # Initialize model
    try:
        from model.architecture import VoiceClassifier
        # TODO: Update input_size based on your feature dimensions
        model = VoiceClassifier(input_size=128, num_classes=3)
        model = model.to(device)
    except ImportError:
        raise ImportError(
            "Model architecture not found. Please create model/architecture.py "
            "with a VoiceClassifier class."
        )
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create save directory if it doesn't exist
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # Training loop
    best_val_acc = 0.0
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_metrics['loss']:.4f}, "
              f"Train Acc: {train_metrics['accuracy']:.2f}%, "
              f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.2f}%")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), args.save_path)
            print(f"  → Saved best model (val acc: {best_val_acc:.2f}%)")
    
    print("=" * 60)
    print(f"Training complete! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {args.save_path}")


if __name__ == '__main__':
    main()
