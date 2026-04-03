from dataset.grid_dataset import GridDataset

dataset = GridDataset("data/grid/s1", max_videos=5)

print("Dataset size:", len(dataset))

x, y = dataset[0]

print("Sample X shape:", x.shape)
print("Sample Y:", y)