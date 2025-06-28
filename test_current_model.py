import torch
import numpy as np
from models.lip_reader_model_lightweight import LightweightLipReaderModel
from utils.text_utils import TextProcessor

def test_current_model():
    """Test the current trained model to see what it's predicting"""
    print("Testing Current Model Predictions...")
    
    # Load text processor
    text_processor = TextProcessor()
    print(f"Vocabulary size: {text_processor.vocab_size}")
    print(f"Vocabulary: {list(text_processor.char_to_idx.keys())}")
    
    # Load model
    model_path = 'best_lightweight_lip_reader_model.pth'
    try:
        model = LightweightLipReaderModel(text_processor.vocab_size)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        print(f"✅ Model loaded from {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test with random inputs
    print("\n=== Testing Model Predictions ===")
    
    for i in range(5):
        # Create random input (batch_size=1, seq_len=40, channels=3, height=64, width=64)
        dummy_input = torch.randn(1, 40, 3, 64, 64)
        
        with torch.no_grad():
            logits = model(dummy_input)
            predictions = torch.argmax(logits, dim=-1)
            
            # Decode predictions
            decoded_text = text_processor.tensor_to_text(predictions[0])
            
            # Check prediction diversity
            unique_preds = torch.unique(predictions[0])
            
            print(f"Sample {i+1}:")
            print(f"  Predicted: '{decoded_text}'")
            print(f"  Unique tokens: {unique_preds.tolist()}")
            print(f"  Diversity: {len(unique_preds)} unique tokens")
            print()
    
    # Check if model is predicting mostly the same token
    print("=== Prediction Analysis ===")
    all_predictions = []
    for i in range(10):
        dummy_input = torch.randn(1, 40, 3, 64, 64)
        with torch.no_grad():
            logits = model(dummy_input)
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions[0].tolist())
    
    # Count token frequencies
    from collections import Counter
    token_counts = Counter(all_predictions)
    
    print("Token frequency distribution:")
    for token_idx, count in token_counts.most_common(10):
        token = text_processor.idx_to_char.get(token_idx, f"UNK_{token_idx}")
        percentage = (count / len(all_predictions)) * 100
        print(f"  {token}: {count} times ({percentage:.1f}%)")
    
    # Check if model is stuck on one token
    most_common_token = token_counts.most_common(1)[0]
    most_common_percentage = (most_common_token[1] / len(all_predictions)) * 100
    
    if most_common_percentage > 80:
        print(f"\n❌ PROBLEM: Model is stuck on token '{text_processor.idx_to_char.get(most_common_token[0], f'UNK_{most_common_token[0]}')}' ({most_common_percentage:.1f}% of predictions)")
        print("This indicates the model is not learning properly!")
    else:
        print(f"\n✅ Model shows reasonable prediction diversity")

if __name__ == "__main__":
    test_current_model() 