import torch
from models.lip_reader_model_lightweight import LightweightLipReaderModel
from utils.text_utils import TextProcessor

def test_lightweight_model():
    """Test the lightweight model with dummy data"""
    print("Testing Lightweight Lip Reader Model...")
    
    # Initialize text processor
    text_processor = TextProcessor()
    num_classes = text_processor.vocab_size
    print(f"Vocabulary size: {num_classes}")
    
    # Create model
    model = LightweightLipReaderModel(num_classes)
    model.gradient_checkpointing_enable()
    
    # Test with different input sizes
    test_configs = [
        (1, 30, 64, 64),  # batch_size, frames, height, width
        (2, 40, 64, 64),
        (1, 50, 64, 64),
    ]
    
    for batch_size, frames, height, width in test_configs:
        print(f"\nTesting with batch_size={batch_size}, frames={frames}, size={height}x{width}")
        
        # Create dummy input
        x = torch.randn(batch_size, frames, 3, height, width)
        
        try:
            # Forward pass
            with torch.no_grad():
                output = model(x)
            
            print(f"✅ Success! Output shape: {output.shape}")
            
            # Test backward pass
            loss = output.sum()
            loss.backward()
            print(f"✅ Backward pass successful!")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    print("\n🎉 All tests passed!")
    return True

if __name__ == "__main__":
    test_lightweight_model() 