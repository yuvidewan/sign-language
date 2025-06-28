import torch
from utils.text_utils import TextProcessor
from models.lip_reader_model_lightweight import LightweightLipReaderModel

def test_text_processing():
    """Test text processing and model predictions"""
    print("=== Testing Text Processing ===")
    
    # Initialize text processor
    text_processor = TextProcessor()
    
    print(f"Vocabulary size: {text_processor.vocab_size}")
    print(f"Character to index: {text_processor.char_to_idx}")
    print(f"Index to character: {text_processor.idx_to_char}")
    
    # Test text conversion
    test_text = "hello world"
    print(f"\nTest text: '{test_text}'")
    
    # Convert to tensor
    tensor = text_processor.text_to_tensor(test_text)
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor values: {tensor.tolist()}")
    
    # Convert back to text
    decoded_text = text_processor.tensor_to_text(tensor)
    print(f"Decoded text: '{decoded_text}'")
    
    # Test model predictions
    print("\n=== Testing Model Predictions ===")
    
    # Load model
    model_path = 'best_lightweight_lip_reader_model.pth'
    model = LightweightLipReaderModel(text_processor.vocab_size)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 40, 3, 64, 64)  # (batch, seq, channels, height, width)
    
    # Get predictions
    with torch.no_grad():
        logits = model(dummy_input)
        predictions = torch.argmax(logits, dim=-1)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[0][:20].tolist()}")  # First 20 predictions
    
    # Decode predictions
    decoded_predictions = text_processor.tensor_to_text(predictions[0])
    print(f"Decoded predictions: '{decoded_predictions}'")
    
    # Test CTC decoding
    print("\n=== Testing CTC Decoding ===")
    ctc_decoded = text_processor.decode_ctc(logits)
    print(f"CTC decoded: {ctc_decoded}")
    
    # Check what characters are being predicted
    print("\n=== Character Analysis ===")
    unique_chars = set()
    for idx in predictions[0]:
        if idx < len(text_processor.idx_to_char):
            char = text_processor.idx_to_char[idx.item()]
            unique_chars.add(char)
    
    print(f"Unique characters predicted: {sorted(unique_chars)}")
    
    # Count character frequencies
    char_counts = {}
    for idx in predictions[0]:
        if idx < len(text_processor.idx_to_char):
            char = text_processor.idx_to_char[idx.item()]
            char_counts[char] = char_counts.get(char, 0) + 1
    
    print("Character frequencies:")
    for char, count in sorted(char_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  '{char}': {count}")

if __name__ == "__main__":
    test_text_processing() 