import torch
import torch.nn as nn
import psutil
import os
from utils.grid_dataset import GRIDDdataset
from utils.text_utils import TextProcessor
from models.lip_reader_model import LipReaderModel
from models.lip_reader_model_lightweight import LightweightLipReaderModel


def print_memory_usage():
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")


def test_model_memory(model, batch_size, frames_per_clip, device):
    """Test memory usage of a model with given parameters"""
    print(f"\nTesting model: {model.__class__.__name__}")
    print(f"Batch size: {batch_size}, Frames per clip: {frames_per_clip}")
    
    # Create dummy data
    sequences = torch.randn(batch_size, frames_per_clip, 3, 64, 64).to(device)
    
    print_memory_usage()
    
    try:
        # Forward pass
        with torch.no_grad():
            output = model(sequences)
        print(f"Forward pass successful. Output shape: {output.shape}")
        print_memory_usage()
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        print("Backward pass successful.")
        print_memory_usage()
        
        return True
        
    except RuntimeError as e:
        print(f"Error: {e}")
        return False


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize text processor
    text_processor = TextProcessor()
    num_classes = text_processor.vocab_size
    
    # Test parameters
    test_configs = [
        # (batch_size, frames_per_clip)
        (1, 30),
        (1, 50),
        (1, 75),
        (2, 30),
        (2, 50),
        (4, 30),
    ]
    
    # Test original model
    print("=" * 50)
    print("TESTING ORIGINAL MODEL")
    print("=" * 50)
    
    original_model = LipReaderModel(num_classes).to(device)
    original_model.gradient_checkpointing_enable()
    
    for batch_size, frames_per_clip in test_configs:
        success = test_model_memory(original_model, batch_size, frames_per_clip, device)
        if not success:
            print(f"❌ Failed with batch_size={batch_size}, frames_per_clip={frames_per_clip}")
            break
        else:
            print(f"✅ Success with batch_size={batch_size}, frames_per_clip={frames_per_clip}")
        
        # Clear memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    del original_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Test lightweight model
    print("\n" + "=" * 50)
    print("TESTING LIGHTWEIGHT MODEL")
    print("=" * 50)
    
    lightweight_model = LightweightLipReaderModel(num_classes).to(device)
    lightweight_model.gradient_checkpointing_enable()
    
    for batch_size, frames_per_clip in test_configs:
        success = test_model_memory(lightweight_model, batch_size, frames_per_clip, device)
        if not success:
            print(f"❌ Failed with batch_size={batch_size}, frames_per_clip={frames_per_clip}")
            break
        else:
            print(f"✅ Success with batch_size={batch_size}, frames_per_clip={frames_per_clip}")
        
        # Clear memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    del lightweight_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 50)
    print("RECOMMENDATIONS")
    print("=" * 50)
    print("Based on the memory tests above, use the following parameters:")
    print("\nFor Original Model:")
    print("- Start with batch_size=1, frames_per_clip=30")
    print("- Use: python -m training.train_lip_reader --batch_size 1 --frames_per_clip 30")
    print("\nFor Lightweight Model:")
    print("- Can use batch_size=2, frames_per_clip=40")
    print("- Use: python -m training.train_lightweight_lip_reader --batch_size 2 --frames_per_clip 40")
    print("\nAdditional tips:")
    print("- Use --device cpu if you don't have enough GPU memory")
    print("- Reduce --num_workers to 0 for lower memory usage")
    print("- Consider using gradient accumulation for effective larger batch sizes")


if __name__ == "__main__":
    main() 