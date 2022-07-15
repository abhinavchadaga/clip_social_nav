import os
import pickle
from typing import Tuple
from termcolor import cprint
import torch
import pytorch_lightning as pl
from lidar_helper import get_stack
import numpy as np

from torch.utils.data import Dataset, DataLoader


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
