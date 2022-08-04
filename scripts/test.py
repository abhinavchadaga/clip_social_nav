from model import CLIPSocialNavModel, VisionTransformer, JoyStickEncoder
from dataset import CLIPDataModule

dm = CLIPDataModule(data_path='./data_subset',
                    batch_size=64,
                    num_workers=10,
                    verbose=True)

dm.setup()
batch = next(iter(dm.train_dataloader()))

model = CLIPSocialNavModel(lidar_encoder=VisionTransformer(),
                           joystick_encoder=JoyStickEncoder(output_dim=512))
out = model(*batch)
print(out[0].shape)
print(out[1].shape)