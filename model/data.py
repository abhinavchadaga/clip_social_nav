from os import listdir
from os.path import join
import pickle
import time
from typing import Tuple

from torch.utils.data import Dataset, DataLoader
import numpy as np


class CLIPSet(Dataset):
    """ Dataset object representing the data from a single rosbag
    """

    def __init__(self, data_dir: str, future_joy_len=300) -> None:
        """ Initialize a CLIPSet object,
            save path to data but do not load into RAM

        Args:
            data_dir (str): path to the data pulled from a single rosbag
            future_joy_len (str): number of messages to use from each joystick data point
        """
        super().__init__()
        # save paths to lidar, joystick, and goals data
        self.lidar_dir = join(data_dir, 'lidar')
        self.joystick_dir = join(data_dir, 'joystick')
        self.goals_dir = join(data_dir, 'goals')
        # sanity check
        assert len(listdir(self.lidar_dir)) == len(
            listdir(self.joystick_dir)) == len(listdir(self.goals_dir))
        # save sequence length and future_joy_length
        self.length = len(listdir(self.lidar_dir))
        self.future_joy_len = future_joy_len

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # load lidar stack from pkl file
        path_to_lidar_stack = join(self.lidar_dir, f'{index}.pkl')
        lidar_stack = pickle.load(open(path_to_lidar_stack, 'rb'))

        # load goal data from pkl file
        path_to_goal = join(self.goals_dir, f'{index}.pkl')
        goal = pickle.load(open(path_to_goal, 'rb'))

        # load joystick data from pkl file
        path_to_joystick = join(self.joystick_dir, f'{index}.pkl')
        joystick = pickle.load(open(path_to_joystick, 'rb'))

        # pad joystick sample with zeros if needed
        # delete the linear_y column
        diff = max(0, self.future_joy_len - joystick.shape[0])
        joystick = np.pad(joystick,
                          pad_width=((0, diff), (0, 0)))
        joystick = joystick[:self.future_joy_len, :]
        joystick = np.delete(joystick, 1, axis=1)

        return lidar_stack, goal, joystick
