from datetime import datetime
import torch
from pytorch_lightning.loops import TrainingEpochLoop
from pytorch_lightning import Trainer
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging, ModelCheckpoint

import config as CFG
from data import CLIPDataModule
from encoders import VisionTransformer, JoyStickEncoder
from model import CLIPSocialNavModel

# create Data Module
dm = CLIPDataModule(data_path=CFG.data_path,
                    batch_size=CFG.batch_size,
                    num_workers=CFG.num_workers,
                    joy_len=CFG.joy_len,
                    pin_memory=CFG.pin_memory,
                    use_weighted_sampling=CFG.use_weighted_sampling,
                    verbose=True)

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

# training callbacks
early_stopping_cb = EarlyStopping(monitor='validation_loss',
                                  mode='min',
                                  min_delta=0.00,
                                  patience=10)
swa_cb = StochasticWeightAveraging(swa_lrs=1e-2)
model_checkpoint_cb = ModelCheckpoint(
    dirpath='trained_models/',
    filename=datetime.now().strftime("%d-%m-%Y-%H-%M-%S"),
    monitor='training_loss',
    mode='min')

trainer = Trainer(
    accelerator='gpu',
    logger=pl_loggers.TensorBoardLogger("lightning_logs/clip_social_nav/"),
    callbacks=[model_checkpoint_cb, swa_cb],
    gradient_clip_val=1.0,
    max_epochs=CFG.epochs,
    log_every_n_steps=20)

num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    trainer = Trainer(
        accelerator='gpu',
        devices=8,
        strategy='dp',
        logger=pl_loggers.TensorBoardLogger("lightning_logs/clip_social_nav/"),
        callbacks=[model_checkpoint_cb, swa_cb],
        gradient_clip_val=1.0,
        max_epochs=CFG.epochs,
        log_every_n_steps=20)
else:
    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        logger=pl_loggers.TensorBoardLogger("lightning_logs/clip_social_nav/"),
        callbacks=[model_checkpoint_cb, swa_cb],
        gradient_clip_val=1.0,
        max_epochs=CFG.epochs,
        log_every_n_steps=20)

# fit model
trainer.fit(model, dm)

# save model
torch.save(
    model.state_dict(),
    'trained_models/' + datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + '.pt')

print('Model has been trained and saved')