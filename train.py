from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging
import tensorboard as tb

from dataset import CLIPDataModule
from model import CLIPSocialNavModel, VisionTransformer, JoyStickEncoder

# lidar encoder parameters
IMAGE_SIZE = 100
INPUT_CHANNELS = 10
PATCH_SIZE = 4
EMBEDDING_SIZE = 256
NHEAD = 8
DIM_FEEDFORWARD = 2048
LE_DROPOUT = 0.2
NUM_LAYERS = 4

# joystick encoder parameters
JOY_LEN = 300

# model parameters
OUTPUT_DIM = 128
TEMPERATURE = 0.07

# hyperparameters
BATCH_SIZE = 128
MAX_EPOCHS = 23
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.2
PIN_MEMORY = True
WARMUP = 2000
MAX_ITERS = 1000

# get data
dm = CLIPDataModule(data_path='./data',
                    batch_size=BATCH_SIZE,
                    joy_len=JOY_LEN,
                    num_workers=8,
                    pin_memory=PIN_MEMORY,
                    use_weighted_sampling=False,
                    verbose=True)

# initialize model
model = CLIPSocialNavModel(
    lidar_encoder=VisionTransformer(img_size=IMAGE_SIZE,
                                    patch_size=PATCH_SIZE,
                                    input_channels=INPUT_CHANNELS,
                                    output_dim=OUTPUT_DIM,
                                    embed_dim=EMBEDDING_SIZE,
                                    depth=NUM_LAYERS,
                                    num_heads=NHEAD,
                                    attn_drop_rate=LE_DROPOUT,
                                    drop_rate=LE_DROPOUT),
    joystick_encoder=JoyStickEncoder(joy_len=JOY_LEN,
                                     output_dim=OUTPUT_DIM),
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
    monitor='training_loss',
    mode='min')

trainer = pl.Trainer(
    accelerator='gpu',
    logger=pl_loggers.TensorBoardLogger("lightning_logs/clip_social_nav/"),
    callbacks=[model_checkpoint_cb, swa_cb, early_stopping_cb],
    gradient_clip_val=1.0,
    precision=16,
    limit_train_batches=0.80,
    limit_val_batches=0.80,
    max_epochs=MAX_EPOCHS,
    log_every_n_steps=20)

trainer.fit(model, dm)

# save model
torch.save(
    model.state_dict(), 'models/clip_social_nav/' +
    datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + '.pt')

print('Model has been trained and saved')
