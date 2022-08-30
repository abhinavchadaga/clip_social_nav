import torch

data_path = '/scratch/abhinavc/clip_social_nav/data'

# dataloader config
batch_size = 32
num_workers = 10
pin_memory = True
use_weighted_sampling = True
joy_len = 250

# lidar encoder config
img_size = 240
patch_size = 8
input_channels = 5
embed_dim = 900
depth = 6
num_heads = 8
drop_rate = 0.1
attn_drop_rate = 0.1

# both encoders
output_dim = 128

# optimizer config
learning_rate = 5e-4
weight_decay = 0.02
temperature = 0.07
patience = 2
factor = 0.5

# training parameters
epochs = 50
sim_threshold = 8.0

# gpu vs cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
