from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import tensorboard as tb

from dataset import CLIPDataModule
from model import CLIPSocialNavModel
from encoders import JoyStickEncoder, LidarEncoder

# lidar encoder parameters
PATCH_SIZE = 16
EMBEDDING_SIZE = 128
NHEAD = 1
LE_DROPOUT = 0.1
NUM_LAYERS = 3

# joystick encoder parameters
JOY_LEN = 300
DROPOUT = 0.1

# model parameters
OUTPUT_DIM = 128
TEMPERATURE = 0.07

# hyperparameters
BATCH_SIZE = 128
MAX_EPOCHS = 100
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-5
FEATURE_SIZE = 100
PIN_MEMORY = False

# get data
dm = CLIPDataModule(data_path='./data',
                    batch_size=64,
                    num_workers=10,
                    pin_memory=PIN_MEMORY,
                    verbose=True)

# initialize model
model = CLIPSocialNavModel(
    lidar_encoder=LidarEncoder(patch_size=PATCH_SIZE,
                               embedding_size=EMBEDDING_SIZE,
                               nhead=NHEAD,
                               dropout=LE_DROPOUT,
                               num_layers=NUM_LAYERS,
                               output_dim=OUTPUT_DIM),
    joystick_encoder=JoyStickEncoder(joy_len=JOY_LEN, output_dim=OUTPUT_DIM),
    temperature=TEMPERATURE)

early_stopping_cb = EarlyStopping(monitor='validation_loss',
                                  mode='min',
                                  min_delta=0.00,
                                  patience=10)

model_checkpoint_cb = ModelCheckpoint(
    dirpath='models/clip_social_nav/',
    filename=datetime.now().strftime("%d-%m-%Y-%H-%M-%S"),
    monitor='validation_loss',
    mode='min')

trainer = pl.Trainer(
    accelerator='gpu',
    devices=2,
    logger=pl_loggers.TensorBoardLogger("lightning_logs/clip_social_nav/"),
    callbacks=[early_stopping_cb, model_checkpoint_cb],
    precision=16,
    strategy='ddp',
    max_epochs=MAX_EPOCHS)

trainer.fit(model, dm)
