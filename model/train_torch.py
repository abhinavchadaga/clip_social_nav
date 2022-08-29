import os
import glob
from datetime import datetime

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from termcolor import cprint
from numerize.numerize import numerize
from tqdm.auto import tqdm

import config as CFG
from data import create_dataloader
from encoders import VisionTransformer, JoyStickEncoder
from model import CLIPSocialNavModel, CLIPLoss


def data_dir_len(data_dir):
    lidar_dir = os.path.join(data_dir, 'lidar')
    return len(os.listdir(lidar_dir))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_batch_similarity(joystick_batch: torch.Tensor) -> float:
    m = torch.empty((joystick_batch.shape[0], joystick_batch.shape[0]))
    joystick_batch = joystick_batch.flatten(1)
    for x in range(m.shape[0]):
        x_1 = joystick_batch[x, :]
        for y in range(m.shape[0]):
            x_2 = joystick_batch[y, :]
            m[x, y] = torch.linalg.norm(x_2 - x_1)
    similarity = torch.sum(m).item() / m.shape[0]**2
    return similarity


# load data directories
data_dirs = glob.glob(os.path.join(CFG.data_path, '*'))
data_dirs = [d for d in data_dirs if os.path.isdir(d)]
data_dirs = sorted(data_dirs, key=data_dir_len, reverse=True)

cprint('loading data from: ', 'white', attrs=['bold'])
for d in data_dirs:
    print(d)
print()

# split data directories into
# train and validation split
train_dirs = data_dirs[:int(0.5 * len(data_dirs))]
val_dirs = data_dirs[int(0.5 * len(data_dirs)):]

cprint('training datasets: ', 'white', attrs=['bold'])
for d in train_dirs:
    print(d)
print()

cprint('validation datasets: ', 'white', attrs=['bold'])
for d in val_dirs:
    print(d)
print()

# create training dataloader
train_loader, trainset_len = create_dataloader(
    data_dirs=train_dirs,
    batch_size=CFG.batch_size,
    num_workers=CFG.num_workers,
    pin_memory=CFG.pin_memory,
    use_weighted_sampling=CFG.use_weighted_sampling,
    train=True,
    joy_len=CFG.joy_len)

# create validation dataloader
val_loader, valset_len = create_dataloader(
    data_dirs=val_dirs,
    batch_size=CFG.batch_size,
    num_workers=CFG.num_workers,
    pin_memory=CFG.pin_memory,
    use_weighted_sampling=CFG.use_weighted_sampling,
    train=False,
    joy_len=CFG.joy_len)

# split size information
cprint(f'training set size:\t {trainset_len}', 'white', attrs=['bold'])
cprint(f'validation set size:\t {valset_len}\n', 'white', attrs=['bold'])

# initialize encoders and model
lidar_encoder = VisionTransformer(img_size=CFG.img_size,
                                  patch_size=CFG.patch_size,
                                  input_channels=CFG.input_channels,
                                  embed_dim=CFG.embed_dim,
                                  output_dim=CFG.output_dim,
                                  depth=CFG.depth,
                                  num_heads=CFG.num_heads,
                                  drop_rate=CFG.drop_rate,
                                  attn_drop_rate=CFG.attn_drop_rate)

joystick_encoder = JoyStickEncoder(joy_len=CFG.joy_len,
                                   output_dim=CFG.output_dim)
model = CLIPSocialNavModel(lidar_encoder=lidar_encoder,
                           joystick_encoder=joystick_encoder)

# loss function and optimizer
loss_fn = CLIPLoss(temperature=CFG.temperature)
# modify only the weights and not biases
weights = [
    param for name, param in model.named_parameters() if 'bias' not in name
]

optimizer = optim.AdamW(params=weights,
                        lr=CFG.learning_rate,
                        weight_decay=CFG.weight_decay,
                        amsgrad=True)

cprint(f'using {CFG.device}\n', 'cyan', attrs=['bold'])
model = model.to(CFG.device)
loss_fn = loss_fn.to(CFG.device)

cprint('MODEL PARAMETERS: ', 'white', attrs=['bold'])
lidar_param_count = count_parameters(lidar_encoder)
joystick_param_count = count_parameters(joystick_encoder)
cprint(f'lidar_encoder:\t\t{numerize(lidar_param_count)}', 'white')
cprint(f'joystick_encoder:\t{numerize(joystick_param_count)}', 'white')
cprint(f'total:\t\t\t{numerize(lidar_param_count + joystick_param_count)}',
       'white')
print('\n')

cprint(f'batch size:\t {CFG.batch_size}', 'white', attrs=['bold'])
cprint(f'learning rate:\t {CFG.learning_rate}', 'white', attrs=['bold'])


def train_one_epoch(epoch_index, tb_writer):
    with tqdm(train_loader, total=len(train_loader), unit='batch') as tepoch:
        for lidar, goal, joystick in tepoch:
            tepoch.set_description(f'epoch {epoch_index + 1}')
            # move tensors to gpu
            lidar, joystick, goal = lidar.to(CFG.device), joystick.to(
                CFG.device), goal.to(CFG.device)

            # measure batch similarity
            joy_sim = get_batch_similarity(joystick)
            # skip batch if joystick batch
            # not diverse enough
            if joy_sim <= CFG.sim_threshold:
                continue

            # reset gradient
            optimizer.zero_grad()
            # forward pass
            lidar_features, joystick_features = model(lidar, goal, joystick)
            # calculate loss and gradients
            loss = loss_fn(lidar_features, joystick_features)
            loss.backward()
            # change weights
            optimizer.step()
            # scheduler step
            tepoch.set_postfix(loss=loss.item())


def validate_one_epoch(epoch_index):
    pass


for i in range(5):
    train_one_epoch(i, None)
