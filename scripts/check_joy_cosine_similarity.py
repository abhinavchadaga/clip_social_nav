from dataset import CLIPDataModule
import torch
from torch import nn

# define batch size
batch_size = 64

# load and setup data
dm_ns = CLIPDataModule(
    data_path="/home/abhinavchadaga/CS/clip_social_nav/data/",
    batch_size=batch_size,
    num_workers=2,
    verbose=True,
)

dm_ns.setup()

dm = CLIPDataModule(
    data_path='/home/abhinavchadaga/CS/clip_social_nav/data/',
    use_weighted_sampling=True,
    batch_size=batch_size,
    num_workers=2,
    verbose=True
)

dm.setup()

# load a batch
batch = next(iter(dm_ns.train_dataloader()))
joystick = batch[1].flatten(1)
print(joystick.shape)

# cosine similarity
cos = nn.CosineSimilarity(dim=0)

# create similarity matrix
similarity_matrix = torch.empty((batch_size, batch_size))

for x in range(joystick.shape[0]):
    x_1 = joystick[x, :]
    for y in range(joystick.shape[0]):
        x_2 = joystick[y, :]
        similarity_matrix[x, y] = cos(x_1, x_2)

similarity = torch.sum(similarity_matrix).item() / batch_size**2
print(f"similarity: {similarity:.2f}")
