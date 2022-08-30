import os
import pickle
from typing import List, Optional, Tuple
import glob

from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler, DataLoader
import pytorch_lightning as pl
import numpy as np
from termcolor import cprint

import config as CFG


class CLIPSet(Dataset):
    """ Dataset object representing the data from a single rosbag
    """

    def __init__(self, data_dir: str, joy_len=200) -> None:
        """ initialize a CLIPSet object,
            save path to data but do not load into RAM

        Args:
            data_dir (str): path to the data pulled from a single rosbag
            future_joy_len (str): number of messages to use from each joystick data point
        """
        super().__init__()
        # save paths to lidar, joystick, goals and weights data
        self.lidar_dir = os.path.join(data_dir, 'lidar')
        self.joystick_dir = os.path.os.path.join(data_dir, 'joystick')
        self.goals_dir = os.path.join(data_dir, 'goals')
        self.weights_path = os.path.join(data_dir, 'weights/weights.pkl')
        # sanity check
        if not (len(os.listdir(self.lidar_dir)) == len(
                os.listdir(self.joystick_dir)) == len(
                    os.listdir(self.goals_dir))):
            raise Exception(data_dir)
        # save sequence length and future_joy_length
        self.length = len(os.listdir(self.lidar_dir))
        self.future_joy_len = joy_len

    def __len__(self) -> int:
        """ return the length of the of dataset
        """
        return self.length

    def __getitem__(self,
                    index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ return the sample at dataset[index]

        Args:
            index (int): index of sample to return

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: lidar, goal, joystick
                lidar.shape: (batch_size, num_channels, img_size, img_size)
                goal.shape: (batch_size, 2)
                joystick.shape: (batch_size, joy_len, 2)
        """
        # load lidar stack from pkl file
        path_to_lidar_stack = os.path.join(self.lidar_dir, f'{index}.pkl')
        lidar_stack = pickle.load(open(path_to_lidar_stack, 'rb'))

        # load goal data from pkl file
        path_to_goal = os.path.join(self.goals_dir, f'{index}.pkl')
        goal = pickle.load(open(path_to_goal, 'rb'))

        # load joystick data from pkl file
        path_to_joystick = os.path.join(self.joystick_dir, f'{index}.pkl')
        joystick = pickle.load(open(path_to_joystick, 'rb'))

        # pad joystick sample with zeros if needed
        # delete the linear_y column
        diff = max(0, self.future_joy_len - joystick.shape[0])
        joystick = np.pad(joystick, pad_width=((0, diff), (0, 0)))
        joystick = joystick[:self.future_joy_len, :]
        joystick = np.delete(joystick, 1, axis=1)

        return lidar_stack, goal, joystick

    def load_weights(self):
        weights = pickle.load(open(self.weights_path, 'rb'))
        return weights


class CLIPDataModule(pl.LightningDataModule):

    def __init__(self,
                 data_path: str,
                 batch_size: int,
                 num_workers: int,
                 joy_len=200,
                 pin_memory=False,
                 use_weighted_sampling=False,
                 verbose=False) -> None:
        super().__init__()
        # instance variables
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.joy_len = joy_len
        self.pin_memory = pin_memory
        self.use_weighted_sampling = use_weighted_sampling
        self.verbose = verbose

        # load data directories
        data_dirs = glob.glob(os.path.os.path.join(data_path, "*"))
        data_dirs = [d for d in data_dirs if os.path.isdir(d)]
        data_dirs = sorted(data_dirs, key=self._data_len, reverse=True)
        # information
        if self.verbose:
            cprint('loading data from: ', 'white', attrs=['bold'])
            for d in data_dirs:
                print(d)
            print()

        # setup train test split
        self.train_dirs = data_dirs[:int(0.5 * len(data_dirs))]
        self.val_dirs = data_dirs[int(0.5 * len(data_dirs)):]
        # information
        if self.verbose:
            cprint('training datasets: ', 'white', attrs=['bold'])
            for d in self.train_dirs:
                print(d)
            print()

            cprint('validation datasets: ', 'white', attrs=['bold'])
            for d in self.val_dirs:
                print(d)
            print()

    def _data_len(self, data_dir: str) -> int:
        """ return the length of data from a data directory (one rosbag of data)
        """
        lidar_dir = os.path.join(data_dir, 'lidar')
        return len(os.listdir(lidar_dir))

    def _concatenate_dataset(
            self, data_dirs: str) -> Tuple[ConcatDataset, List[float]]:
        tmp_sets = []
        weights = []
        for d in data_dirs:
            tmp = CLIPSet(data_dir=d, joy_len=self.joy_len)
            tmp_sets.append(tmp)
            if self.use_weighted_sampling:
                tmp_weights = tmp.load_weights()
                weights.extend(tmp_weights)

        return ConcatDataset(tmp_sets), weights

    def setup(self, stage: Optional[str] = None) -> None:
        """ create training and validation sets

        Args:
            stage (Optional[str]): one of either ['fit', 'validate', 'test', 'predict']
        """

        if stage in (None, "fit"):
            self.train_set, self.train_weights = self._concatenate_dataset(
                data_dirs=self.train_dirs)
            self.val_set, self.val_weights = self._concatenate_dataset(
                data_dirs=self.val_dirs)

            if self.verbose:
                cprint(f'training set size:\t {len(self.train_set)}',
                       'white',
                       attrs=['bold'])
                cprint(f'validation set size:\t {len(self.val_set)}\n',
                       'white',
                       attrs=['bold'])

    def train_dataloader(self) -> DataLoader:
        """ return the training dataloader
        """
        if self.use_weighted_sampling:
            sampler = WeightedRandomSampler(weights=self.train_weights,
                                            num_samples=len(
                                                self.train_weights),
                                            replacement=False)
            return DataLoader(dataset=self.train_set,
                              batch_size=self.batch_size,
                              sampler=sampler,
                              num_workers=self.num_workers,
                              pin_memory=self.pin_memory,
                              drop_last=True)

        return DataLoader(dataset=self.train_set,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=True)

    def val_dataloader(self) -> DataLoader:
        """ return validation dataloader
        """
        if self.use_weighted_sampling:
            sampler = WeightedRandomSampler(weights=self.val_weights,
                                            num_samples=len(self.val_weights),
                                            replacement=False)
            return DataLoader(dataset=self.val_set,
                              batch_size=self.batch_size,
                              sampler=sampler,
                              num_workers=self.num_workers,
                              pin_memory=self.pin_memory,
                              drop_last=True)

        return DataLoader(dataset=self.val_set,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=True)


if __name__ == "__main__":
    # test the dataloader
    dm = CLIPDataModule(data_path=CFG.data_path,
                        batch_size=64,
                        num_workers=8,
                        joy_len=200,
                        use_weighted_sampling=True,
                        verbose=True)
    dm.setup()

    for i, (lidar, goal, joystick) in enumerate(dm.train_dataloader()):
        if i == 0:
            print(lidar.shape)
            print(goal.shape)
            print(joystick.shape)
    cprint('successfully loaded all training batches',
           color='green',
           attrs=['bold'])

    for i, (lidar, goal, joystick) in enumerate(dm.val_dataloader()):
        pass

    cprint('successfully loaded all validation batches',
           'green',
           attrs=['bold'])
