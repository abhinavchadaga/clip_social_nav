from datetime import datetime
from matplotlib.animation import ImageMagickBase

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging
import tensorboard as tb

from dataset import CLIPDataModule
from model import CLIPSocialNavModel, VisionTransformer, JoyStickEncoder
from my_vision_transformer import LidarEncoder
from vit_lucidrains import ViT

# lidar encoder parameters
IMAGE_SIZE = 100
INPUT_CHANNELS = 10
PATCH_SIZE = 4
EMBEDDING_SIZE = 256
NHEAD = 4
DIM_FEEDFORWARD = 2048
LE_DROPOUT = 0.1
NUM_LAYERS = 2

# joystick encoder parameters
JOY_LEN = 300
DROPOUT = 0.1

# model parameters
OUTPUT_DIM = 100
TEMPERATURE = 0.07

# hyperparameters
BATCH_SIZE = 1024
MAX_EPOCHS = 32
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.2
PIN_MEMORY = True
WARMUP = 2000
MAX_ITERS = 1000

# get data
dm = CLIPDataModule(data_path='./data',
                    batch_size=BATCH_SIZE,
                    joy_len=JOY_LEN,
                    num_workers=10,
                    pin_memory=PIN_MEMORY,
                    verbose=True)

vision_transformer = VisionTransformer(img_size=IMAGE_SIZE,
                                       patch_size=PATCH_SIZE,
                                       input_channels=INPUT_CHANNELS,
                                       output_dim=OUTPUT_DIM,
                                       embed_dim=EMBEDDING_SIZE,
                                       depth=NUM_LAYERS,
                                       num_heads=NHEAD,
                                       attn_drop_rate=LE_DROPOUT,
                                       drop_rate=LE_DROPOUT)
# initialize model
model = CLIPSocialNavModel(lidar_encoder=vision_transformer,
                           joystick_encoder=JoyStickEncoder(
                               joy_len=JOY_LEN, output_dim=OUTPUT_DIM),
                           temperature=TEMPERATURE,
                           lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY,
                           warmup=WARMUP,
                           max_iters=MAX_ITERS)

early_stopping_cb = EarlyStopping(monitor='validation_loss',
                                  mode='min',
                                  min_delta=0.00,
                                  patience=10)

swa_cb = StochasticWeightAveraging(swa_lrs=1e-2)

model_checkpoint_cb = ModelCheckpoint(
    dirpath='models/clip_social_nav/',
    filename=datetime.now().strftime("%d-%m-%Y-%H-%M-%S"),
    monitor='validation_loss',
    mode='min')

trainer = pl.Trainer(
    accelerator='gpu',
    strategy='dp',
    devices=8,
    logger=pl_loggers.TensorBoardLogger("lightning_logs/clip_social_nav/"),
    callbacks=[model_checkpoint_cb, swa_cb],
    gradient_clip_val=1.0,
    precision=16,
    replace_sampler_ddp=False,
    sync_batchnorm=True,
    max_epochs=MAX_EPOCHS,
    num_sanity_val_steps=0,
    log_every_n_steps=20)

trainer.fit(model, dm)

# save model
torch.save(
    model.state_dict(), 'models/clip_social_nav/' +
    datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + '.pt')

print('Model has been trained and saved')