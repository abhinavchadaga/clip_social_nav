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
from data import CLIPDataModule
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


# create Data Module
dm = CLIPDataModule(data_path=CFG.data_path,
                    batch_size=CFG.batch_size,
                    num_workers=CFG.num_workers,
                    joy_len=CFG.joy_len,
                    pin_memory=CFG.pin_memory,
                    use_weighted_sampling=CFG.use_weighted_sampling,
                    verbose=True)

# split size information
cprint(f'training set size:\t {len(dm.train_set)}', 'white', attrs=['bold'])
cprint(f'validation set size:\t {len(dm.val_set)}\n', 'white', attrs=['bold'])

# encoders and model
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
                           joystick_encoder=joystick_encoder,
                           temperature=CFG.temperature)

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
    with tqdm(dm.train_dataloader(),
              total=len(dm.train_dataloader()),
              unit='batch') as tepoch:
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
            tepoch.set_postfix(training_step_loss=loss.item())


def validate_one_epoch(epoch_index):
    pass


for i in range(5):
    train_one_epoch(i, None)
