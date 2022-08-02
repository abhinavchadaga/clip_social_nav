import glob
import os
import pickle
import random
from typing import Optional, Tuple

import numpy as np
import pytorch_lightning as pl
from termcolor import cprint
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm.auto import tqdm

from lidar_helper import get_stack, STACK_LEN


class CLIPSet(Dataset):
    """ data from one rosbag
    """
    def __init__(self,
                 pickle_file_path: str,
                 ignore_first_n=30,
                 future_joy_len=300,
                 include_lidar_file_names=False) -> None:
        """ Initialize a pytorch dataset using a single rosbag

        Args:
            pickle_file_path (str): path to the pkl file, infer the path to the lidar data
            ignore_first (int, optional): number of initial frames to skip. Defaults to 30.
            joy_pred_len (int, optional): number of joystick data points to use from timestamp t.
                Defaults to 300.
            include_lidar_file_names (bool, optional): return file names with lidar imgs.
                Defaults to False.
        """

        # check if the pickle file path exists
        # check if the pickle file is processed using file suffix
        if not os.path.exists(
                pickle_file_path.replace('_data.pkl', '_final.pkl')):
            raise Exception(
                "Pickle file does not exist. Please process the pickle file first.."
            )

        with open(pickle_file_path.replace('_data.pkl', '_final.pkl'),
                  'rb') as f:
            self.data = pickle.load(f)

        # 3 dimensional matrix of future joystick values at each time index
        self.future_joy_data = self.data['future_joystick']
        self.joy_pred_len = future_joy_len

        # get lidar image information
        self.lidar_dir: str = pickle_file_path[:-4]
        self.lidar_img_paths = os.listdir(self.lidar_dir)

        # set ignore_first frame to be at least 30
        self.ignore_first = ignore_first_n if ignore_first_n > 30 else 30

        # toggle visualization for bev lidar images
        self.include_lidar_file_names = include_lidar_file_names

    def __len__(self) -> int:
        """ return length of the dataset

        equal to num_samples - ignore_first_n - STACK_LEN

        Returns:
            int: number of samples in the dataset
        """
        return len(self.lidar_img_paths) - self.ignore_first - STACK_LEN

    def __getitem__(self,
                    index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ get a sample from the dataset at this index

        Args:
            index (int): sample from this index

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: lidar_stack, joystick_data, relative_goal
        """
        # skip the ignore_first frames, and offset by STACK_LEN
        # so that no ignore_first frames are used in any lidar_stack
        index = self.ignore_first + STACK_LEN + index

        # get lidar stack
        lidar_stack = get_stack(odom=self.data['odom'],
                                lidar_img_dir=self.lidar_dir,
                                i=index)

        if not self.include_lidar_file_names:
            lidar_stack = lidar_stack[0]

        # get joystick data
        joystick_data = self.data['future_joystick'][index]
        # pad with zeros if the length of the joystick sample is not long enough
        diff = max(0, self.joy_pred_len - joystick_data.shape[0])
        joystick_data = np.pad(joystick_data, pad_width=((0, diff), (0, 0)))
        joystick_data = joystick_data[:self.joy_pred_len, :]

        # get 10m goal relative to the current position of the robot
        current_traj = self.data['human_expert_odom'][index]
        robot_x, robot_y = current_traj[0][0], current_traj[0][1]
        wf_goal_x, wf_goal_y = current_traj[-1][0], current_traj[-1][1]
        rel_goal_x, rel_goal_y = wf_goal_x - robot_x, wf_goal_y - robot_y
        relative_goal = np.array([rel_goal_x, rel_goal_y], dtype=np.float32)

        return lidar_stack, joystick_data, relative_goal


class CLIPDataModule(pl.LightningDataModule):
    """ Data wrapper for entire dataset

    setup training, validation splits
    """
    def __init__(self,
                 data_path: str,
                 batch_size: int,
                 num_workers=0,
                 ignore_first_n=30,
                 joy_len=300,
                 include_lidar_file_names=False,
                 verbose=False) -> None:
        """ Initialize a pytorch lightning data module from a directory of processed rosbags

        Args:
            data_dir (str): path to the processed rosbag data
            batch_size (int): number of samples to retrieve at once.
            num_workers (int, optional): number of threads to use when loading data. Defaults to 0.
            ignore_first_n (int, optional): number of initial frames of a rosbag to skip.
                Defaults to 30.
            joy_len (int, optional): number of joystick data points to use from timestamp t.
                Defaults to 300.
            include_lidar_file_names (bool, optional): return file names with lidar imgs.
                Defaults to False.
            verbose (bool, optional): print datamodule information . Defaults to False.
        """

        super().__init__()
        self.data_dir = data_path
        if data_path is None or not os.path.exists(data_path):
            raise ValueError("Make sure to pass in a valid data directory")

        self.joy_len = joy_len
        self.dataset = None
        self.training_set = []
        self.validation_set = []
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ignore_first_n = ignore_first_n

        # diagnostics
        self.include_lidar_file_names = include_lidar_file_names
        self.verbose = verbose

        # load the pickle files
        self.pickle_file_paths = glob.glob(
            os.path.join(self.data_dir, '*_final.pkl'))

    def setup(self, stage: Optional[str] = None) -> None:
        """ Setup training and validation splits

        :param stage: one of either ['fit', 'validate', 'test', 'predict']
        """
        if stage in (None, 'fit'):
            # print information
            cprint(f'loading data from {self.data_dir}...',
                   'green',
                   attrs=['bold'])
            if self.verbose:
                cprint(f'skip first {self.ignore_first_n + STACK_LEN} frames',
                       'cyan')
                cprint(f'batch size: {self.batch_size}', 'cyan')
                cprint(f'future joystick length: {self.joy_len}\n', 'cyan')

            # shuffle pkl list for train validation split
            random.shuffle(self.pickle_file_paths)
            train_pkl = self.pickle_file_paths[:int(0.75 *
                                                    len(self.pickle_file_paths)
                                                    )]
            val_pkl = self.pickle_file_paths[int(0.75 *
                                                 len(self.pickle_file_paths)):]

            # load training pkl files
            if self.verbose:
                cprint('creating training set...', 'green')
            for pfp in tqdm(train_pkl, position=0, leave=True):
                tmp_dataset = CLIPSet(
                    pickle_file_path=pfp,
                    ignore_first_n=self.ignore_first_n,
                    future_joy_len=self.joy_len,
                    include_lidar_file_names=self.include_lidar_file_names)

                self.training_set.append(tmp_dataset)

            # load validation pkl files
            if self.verbose:
                cprint('creating validation set...', 'green')
            for pfp in tqdm(val_pkl, position=0, leave=True):
                tmp_dataset = CLIPSet(
                    pickle_file_path=pfp,
                    ignore_first_n=self.ignore_first_n,
                    future_joy_len=self.joy_len,
                    include_lidar_file_names=self.include_lidar_file_names)

                self.validation_set.append(tmp_dataset)

            # concat dataset objects into full training and validation set
            self.training_set, self.validation_set = ConcatDataset(
                datasets=self.training_set), ConcatDataset(
                    datasets=self.validation_set)

            # ensure that training set is larger than validation set
            # for small bag counts
            if len(self.training_set) < len(self.validation_set):
                self.training_set, self.validation_set = self.validation_set, self.training_set

            # display information
            if self.verbose:
                cprint(f'training size: {len(self.training_set)} samples',
                       'cyan')
                cprint(f'validation size: {len(self.validation_set)} samples',
                       'cyan')

    def train_dataloader(self) -> DataLoader:
        """ return training dataloader, shuffled

        Returns:
            DataLoader: training DataLoader
        """
        return DataLoader(self.training_set,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          drop_last=True)

    def val_dataloader(self) -> DataLoader:
        """ return validation dataloader, not shuffled

        Returns:
            DataLoader: validation DataLoader
        """
        return DataLoader(self.validation_set,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=True)