from datetime import datetime
from gc import callbacks

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import tensorboard as tb

from dataset import CLIPDataModule
from model import CLIPSocialNavModel

# lidar encoder parameters
patch_size = 16
embedding_size = 128
le_msa_heads = 1
le_layers = 3

# joystick encoder parameters
future_joy_len = 300
dropout = 0.0

# both encoders
output_dim = 128

# model parameters
max_epochs = 100
output_dim = 128
temperature = 0.7

# hyperparameters
batch_size = 128
learning_rate = 3e-5
weight_decay = 1e-5
feature_size = 100

# get data
dm = CLIPDataModule(data_dir='./data',
                    batch_size=batch_size,
                    future_joy_len=future_joy_len,
                    num_workers=10,
                    verbose=True)

# initialize model
model = CLIPSocialNavModel(patch_size=patch_size,
                           embedding_size=embedding_size,
                           nhead=le_msa_heads,
                           le_num_layers=le_layers,
                           output_dim=output_dim,
                           future_joy_len=future_joy_len,
                           j_dropout=dropout,
                           temperature=temperature)

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
    gpus=2,
    logger=pl_loggers.TensorBoardLogger("lightning_logs/clip_social_nav/"),
    callbacks=[early_stopping_cb, model_checkpoint_cb],
    strategy='dp',
    max_epochs=max_epochs)

trainer.fit(model, dm)
