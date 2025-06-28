from utils.grid_dataset import GRIDDdataset
from utils.text_utils import TextProcessor

def check_training_data():
    """Check the training data and vocabulary"""
    print("=== Checking Training Data ===")
    
    # Load dataset
    dataset = GRIDDdataset('data/GRID', speakers=['s1'], frames_per_clip=40)
    print(f"Dataset size: {len(dataset)}")
    
    # Check sample transcripts
    print("\nSample transcripts:")
    for i in range(min(10, len(dataset))):
        transcript = dataset[i][1]
        print(f"  {i+1}: '{transcript}'")
    
    # Collect all unique words
    all_words = set()
    for i in range(min(100, len(dataset))):  # Check first 100 samples
        transcript = dataset[i][1]
        words = transcript.split()
        all_words.update(words)
    
    print(f"\nUnique words found: {sorted(all_words)}")
    print(f"Total unique words: {len(all_words)}")
    
    # Check character-level vocabulary
    all_chars = set()
    for word in all_words:
        all_chars.update(word.lower())
    
    print(f"\nUnique characters found: {sorted(all_chars)}")
    print(f"Total unique characters: {len(all_chars)}")
    
    # Test text processor with actual data
    print("\n=== Testing Text Processor with Real Data ===")
    text_processor = TextProcessor()
    
    for i in range(min(5, len(dataset))):
        transcript = dataset[i][1]
        print(f"\nOriginal: '{transcript}'")
        
        # Convert to tensor
        tensor = text_processor.text_to_tensor(transcript)
        print(f"Tensor: {tensor.tolist()}")
        
        # Convert back
        decoded = text_processor.tensor_to_text(tensor)
        print(f"Decoded: '{decoded}'")

if __name__ == "__main__":
    check_training_data() 