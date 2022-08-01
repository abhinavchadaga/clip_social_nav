from datetime import datetime
from gc import callbacks

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import tensorboard as tb

from dataset import CLIPDataModule
from model import CLIPSocialNavModel

# get data
dm = CLIPDataModule(data_dir='./data',
                    batch_size=256,
                    future_joy_len=500,
                    num_workers=10,
                    verbose=True)

# initialize model
model = CLIPSocialNavModel(l_msa_heads=4,
                           l_num_layers=3,
                           future_joy_len=500,
                           lr=3e-5)

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
    max_epochs=100)

trainer.fit(model, dm)
