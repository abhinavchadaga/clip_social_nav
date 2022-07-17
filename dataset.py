import glob
import os
import pickle
from typing import Optional, Tuple

import numpy as np
import pytorch_lightning as pl
from PIL import ImageShow, Image
from termcolor import cprint
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from tqdm import tqdm

from lidar_helper import get_stack


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
        return len(self.lidar_img_paths) - self.delay

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        # offset index
        index = self.delay + index

        lidar_stack = get_stack(odom=self.data['odom'],
                                lidar_path=self.lidar_dir,
                                i=index)

        if not self.visualize:
            lidar_stack = lidar_stack[0]

        future_joy_data = self.data['future_joystick'][index]
        # TODO: look at way to force same dimension in samples
        # right now limiting to 300 frames / 5 seconds
        future_joy_data = future_joy_data[:300, :]
        return lidar_stack, future_joy_data


class CLIPDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size=int, num_workers=int, delay=30, visualize=False) \
            -> None:
        """Configure Data Module

        args:
            data_dir: path to all the processed pickle files that will make up the dataset
            batch_size: number of lidar stacks and future_joystick samples to pull from
                the dataloader for one training step
            num_workers: threads to use for dataloaders

        """
        super(CLIPDataModule, self).__init__()
        self.dataset = None
        self.validation_set = None
        self.training_set = None
        self.visualize = visualize
        if data_dir is None or not os.path.exists(data_dir):
            raise ValueError("Make sure to pass in a valid data directory")

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.delay = delay

        # load the pickle files
        self.pickle_file_paths = glob.glob(
            os.path.join(data_dir, '*_final.pkl'))

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup training and validation splits

        args:
            stage: one of either ['fit', 'validate', 'test', 'predict']

        """
        if stage in (None, 'fit'):
            datasets = []
            for pfp in tqdm(self.pickle_file_paths):
                tmp_dataset = CLIPSet(pickle_file_path=pfp, delay=self.delay,
                                      visualize=self.visualize)
                datasets.append(tmp_dataset)

            # create full dataset by concatenating datasets
            # generated from every pickle file and lidar dir
            self.dataset = ConcatDataset(datasets=datasets)
            # setup 75-25 training-validation split
            train_size = int(0.75 * len(self.dataset))
            val_size = len(self.dataset) - train_size

            # randomly split into training and validation set
            self.training_set, self.validation_set = random_split(
                dataset=self.dataset, lengths=[train_size, val_size])

    def train_dataloader(self) -> DataLoader:
        # return a pytorch dataloader object
        # with training data
        return DataLoader(self.training_set, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        # return a pytorch dataloader object
        # with validation data
        return DataLoader(self.validation_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, drop_last=True)
