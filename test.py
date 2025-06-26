from utils.grid_dataset import GRIDDdataset

dataset = GRIDDdataset(root_dir='data/GRID/', speakers=['s1'])
frames, transcript = dataset[0]
print("Frames shape:", frames.shape)
print("Transcript:", transcript)