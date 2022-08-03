from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging
import tensorboard as tb

from dataset import CLIPDataModule
from model import CLIPSocialNavModel, ViT, JoyStickEncoder
# from encoders import JoyStickEncoder, LidarEncoder

# lidar encoder parameters
PATCH_SIZE = 64
EMBEDDING_SIZE = 128
NHEAD = 4
LE_DROPOUT = 0.2
NUM_LAYERS = 3

# joystick encoder parameters
JOY_LEN = 300
DROPOUT = 0.2

# model parameters
OUTPUT_DIM = 128
TEMPERATURE = 0.07

# hyperparameters
BATCH_SIZE = 32
MAX_EPOCHS = 32
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-5
FEATURE_SIZE = 100
PIN_MEMORY = False

# get data
dm = CLIPDataModule(data_path='./data',
                    batch_size=BATCH_SIZE,
                    num_workers=10,
                    pin_memory=PIN_MEMORY,
                    verbose=True)

# initialize model
model = CLIPSocialNavModel(lidar_encoder=ViT(image_size=400,
                                             patch_size=16,
                                             num_classes=OUTPUT_DIM,
                                             embedding_size=EMBEDDING_SIZE,
                                             depth=6,
                                             heads=16,
                                             mlp_dim=256,
                                             channels=5,
                                             dropout=0.1,
                                             emb_dropout=0.1,
                                             pool='cls'),
                           joystick_encoder=JoyStickEncoder(
                               joy_len=JOY_LEN, output_dim=OUTPUT_DIM),
                           temperature=TEMPERATURE)

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
    devices=2,
    logger=pl_loggers.TensorBoardLogger("lightning_logs/clip_social_nav/"),
    callbacks=[model_checkpoint_cb, swa_cb],
    strategy="ddp_find_unused_parameters_false",
    gradient_clip_val=1.0,
    replace_sampler_ddp=False,
    max_epochs=MAX_EPOCHS,
    log_every_n_steps=5)

trainer.fit(model, dm)
