import torch
from models.lip_reader_model_lightweight import LightweightLipReaderModel
from utils.text_utils import TextProcessor
from utils.grid_dataset import GRIDDdataset
from torch.utils.data import DataLoader

def verify_training():
    """Verify if the model is actually learning"""
    print("=== Training Verification ===")
    
    # Initialize components
    text_processor = TextProcessor()
    dataset = GRIDDdataset('data/GRID', speakers=['s1'], frames_per_clip=40)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Vocabulary size: {text_processor.vocab_size}")
    
    # Load trained model
    model_path = 'best_lightweight_lip_reader_model.pth'
    model = LightweightLipReaderModel(text_processor.vocab_size)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    print("Model loaded successfully!")
    
    # Test with real data
    print("\n=== Testing with Real Data ===")
    
    for i in range(min(5, len(dataset))):
        frames, transcript = dataset[i]
        
        # Prepare input
        input_tensor = frames.unsqueeze(0)  # Add batch dimension
        
        print(f"\nSample {i+1}:")
        print(f"  Transcript: '{transcript}'")
        print(f"  Input shape: {input_tensor.shape}")
        
        # Get predictions
        with torch.no_grad():
            logits = model(input_tensor)
            predictions = torch.argmax(logits, dim=-1)
        
        print(f"  Logits shape: {logits.shape}")
        print(f"  Predictions shape: {predictions.shape}")
        
        # Decode predictions
        decoded = text_processor.tensor_to_text(predictions[0])
        print(f"  Predicted: '{decoded}'")
        
        # Check prediction distribution
        unique_preds = torch.unique(predictions[0])
        print(f"  Unique predictions: {unique_preds.tolist()}")
        
        # Check if predictions are meaningful
        if len(unique_preds) <= 3:
            print(f"  ⚠️  WARNING: Very few unique predictions - model may not be learning!")
        else:
            print(f"  ✅ Good diversity in predictions")
    
    # Test with random data
    print("\n=== Testing with Random Data ===")
    random_input = torch.randn(1, 40, 3, 64, 64)
    
    with torch.no_grad():
        logits = model(random_input)
        predictions = torch.argmax(logits, dim=-1)
    
    decoded_random = text_processor.tensor_to_text(predictions[0])
    print(f"Random input prediction: '{decoded_random}'")
    
    # Check if model outputs are consistent
    unique_random = torch.unique(predictions[0])
    print(f"Unique random predictions: {unique_random.tolist()}")
    
    if len(unique_random) <= 2:
        print("🚨 CRITICAL: Model is only predicting a few tokens - training may have failed!")
    else:
        print("✅ Model shows some diversity in predictions")

if __name__ == "__main__":
    verify_training() 