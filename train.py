import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from dataset import CLIPDataModule
from model import CLIPSocialNavModel

# get data
dm = CLIPDataModule(data_dir='./data', batch_size=256, num_workers=32)

# initialize model
model = CLIPSocialNavModel()

# create trainer
trainer = pl.Trainer(gpus=2,
                     logger=pl_loggers.TensorBoardLogger(
                         "lightning_logs/clip_social_nav/"),
                     progress_bar_refresh_rate=1, strategy='dp')
# trainer = pl.Trainer()
trainer.fit(model, dm)
