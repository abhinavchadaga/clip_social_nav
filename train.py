import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from dataset import CLIPDataModule
from model import CLIPSocialNavModel

# get data
dm = CLIPDataModule(data_dir='./data', batch_size=32, num_workers=8, verbose=True)

# initialize model
model = CLIPSocialNavModel(l_msa_heads=4, l_num_layers=3)

# create trainer
# trainer = pl.Trainer(gpus=2,
#                      logger=pl_loggers.TensorBoardLogger("lightning_logs/clip_social_nav/"),
#                      progress_bar_refresh_rate=1,
#                      strategy='dp',
#                      max_epochs=1)

trainer = pl.Trainer(accelerator='cpu',
                     max_epochs=1)

trainer.fit(model, dm)
