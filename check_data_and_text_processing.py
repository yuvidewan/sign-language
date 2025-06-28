import torch
from utils.grid_dataset import GRIDDdataset
from utils.text_utils import TextProcessor

if __name__ == "__main__":
    print("Checking data loading and text processing...")
    data_path = 'data/GRID'  # Change if your data is elsewhere
    speakers = ['s1', 's2', 's3']
    frames_per_clip = 40

    dataset = GRIDDdataset(root_dir=data_path, speakers=speakers, frames_per_clip=frames_per_clip)
    print(f"Loaded {len(dataset)} samples.")

    text_processor = TextProcessor()
    print(f"Vocabulary size: {text_processor.vocab_size}")
    print(f"Vocabulary: {list(text_processor.char_to_idx.keys())}")

    # Show a few samples
    for i in range(min(5, len(dataset))):
        frames, transcript = dataset[i]
        print(f"\nSample {i+1}:")
        print(f"  Transcript: '{transcript}'")
        tensor = text_processor.text_to_tensor(transcript)
        print(f"  Encoded: {tensor.tolist()}")
        decoded = text_processor.tensor_to_text(tensor)
        print(f"  Decoded: '{decoded}'")
        print(f"  Frames shape: {frames.shape}")

    print("\nIf the decoded text does not match the transcript, there is a text processing bug!") 