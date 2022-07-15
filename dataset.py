from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import os
import pickle
from typing import Optional, Tuple
from matplotlib import cm
import matplotlib
from termcolor import cprint
import torch
import pytorch_lightning as pl
from tqdm import tqdm
from lidar_helper import get_stack
import numpy as np
import glob
import psutil
import matplotlib.pyplot as plt
from PIL import ImageShow, Image


class CLIPSet(Dataset):
    def __init__(self, pickle_file_path: str, delay=30, visualize=False) -> None:
        """create a CLIPSet object

        capture data from the pickle file
        get path to the bev lidar images but do not load into memory

        args:
            pickle_file_path: path to the processed pickle file
            delay_frame: number of frames to skip from the start,
                automatically set to 30 if argument is less than 
        """

        # check if the pickle file path exists
        # check if the pickle file is processed using file suffix
        if not os.path.exists(pickle_file_path.replace('_data.pkl', '_final.pkl')):
            raise Exception(
                "Pickle file does not exist. Please process the pickle file first..")

        # load pickle file into data attribute
        cprint('Pickle file exists. Loading from pickle file')
        self.data = pickle.load(
            open(pickle_file_path.replace('_data.pkl', '_final.pkl'), 'rb'))

        # get lidar image information
        self.lidar_dir: str = pickle_file_path[:-4]
        self.lidar_img_paths = os.listdir(self.lidar_dir)

        # set delay frame to be at least 30
        self.delay = delay if delay > 30 else 30

        self.visualize = visualize

        cprint('Delay frame is : ' + str(self.delay),
               'yellow', attrs=['bold'])

    def __len__(self) -> int:
        return len(self.lidar_img_paths)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        # offset index
        index = self.delay + index

        lidar_stack = get_stack(odom=self.data['odom'],
                                lidar_path=self.lidar_dir,
                                i=index)

        if not self.visualize:
            lidar_stack = lidar_stack[0]

        future_joy_data = self.data['future_joystick'][index]
        return lidar_stack, future_joy_data


class CLIPDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size=int, num_workers=0, delay=30) -> None:
        super().__init__()
        # check that data directory is valid
        if data_dir is None or not os.path.exists(data_dir):
            raise ValueError("Make sure to pass in a valid data directory")

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.delay = delay
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

        # load the pickle files
        self.pickle_file_paths = glob.glob(
            os.path.join(data_dir, '*_final.pkl'))

        datasets = []
        for pfp in tqdm(self.pickle_file_paths):
            print('path to pickle file: ', pfp)
            tmp_dataset = CLIPSet(pickle_file_path=pfp, delay=self.delay)
            datasets.append(tmp_dataset)

        clipset_full = ConcatDataset(datasets=datasets)
        train_size = int(0.75 * len(clipset_full))
        val_size = len(clipset_full) - train_size
        self.clipset_train, self.clipset_val = random_split(
            clipset_full, [train_size, val_size])

    # def setup(self, stage: Optional[str] = None) -> None:
    #     if stage in (None, 'fit'):
    #         datasets = []
    #         for pfp in tqdm(self.pickle_file_paths):
    #             tmp_dataset = CLIPSet(pickle_file_path=pfp, delay=self.delay)
    #             datasets.append(tmp_dataset)

    #         clipset_full = ConcatDataset(datasets=datasets)
    #         self.clipset_train, self.clipset_val = random_split(
    #             clipset_full, [0.7 * len(clipset_full), 1 - (0.7 * len(clipset_full))])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.clipset_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.clipset_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True)


def main():
    dm = CLIPDataModule(data_dir='./data', batch_size=1)

    train_dataloader = dm.train_dataloader()
    lidar_stack, joystick = next(iter(train_dataloader))
    print(lidar_stack.shape)
    img = Image.fromarray(lidar_stack.numpy().squeeze()[0] * 255)
    ImageShow.show(img)


if __name__ == "__main__":
    main()
