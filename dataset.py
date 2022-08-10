import glob
import os
import pickle
from typing import Optional, Tuple

import numpy as np
import pytorch_lightning as pl
from termcolor import cprint
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler, Subset
from tqdm.auto import tqdm

from utils import STACK_LEN, get_stack


class CLIPSet(Dataset):
    """data from one rosbag"""

    def __init__(self,
                 pickle_file_path: str,
                 ignore_first_n=30,
                 future_joy_len=300,
                 include_lidar_file_names=False, ) -> None:
        """Initialize a pytorch dataset using a single rosbag

        Args:
            pickle_file_path (str): path to the pkl file, infer the path to the lidar data
            ignore_first_n (int, optional): number of initial frames to skip. Defaults to 30.
            future_joy_len (int, optional): number of joystick data points to use from timestamp t.
                Defaults to 300.
            include_lidar_file_names (bool, optional): return file names with lidar imgs.
                Defaults to False.
        """

        # check if the pickle file path exists
        # check if the pickle file is processed using file suffix
        if not os.path.exists(pickle_file_path.replace("_data.pkl", "_final.pkl")):
            raise Exception(
                "Pickle file does not exist. Please process the pickle file first..")

        with open(pickle_file_path.replace("_data.pkl", "_final.pkl"), "rb") as f:
            self.data = pickle.load(f)

        # 3 dimensional matrix of future joystick values at each time index
        self.future_joy_data = self.data["future_joystick"]
        self.joy_pred_len = future_joy_len

        # get lidar image information
        self.lidar_dir: str = pickle_file_path[:-4]
        self.lidar_img_paths = os.listdir(self.lidar_dir)

        # set ignore_first frame to be at least 30
        self.ignore_first_n = ignore_first_n if ignore_first_n > 30 else 30

        # toggle visualization for bev lidar images
        self.include_lidar_file_names = include_lidar_file_names

    def __len__(self) -> int:
        """return length of the dataset

        equal to num_samples - ignore_first_n - STACK_LEN

        Returns:
            int: number of samples in the dataset
        """
        return min(len(self.lidar_img_paths), len(self.data['odom'])) - self.ignore_first_n - STACK_LEN

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """get a sample from the dataset at this index

        Args:
            index (int): index of the sample to load

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: lidar_stack, joystick_data, 10m goal
        """
        # skip the ignore_first_n frames, and offset by STACK_LEN
        # so that no ignore_first_n frames are used in any lidar_stack
        index = self.ignore_first_n + STACK_LEN + index

        # get lidar stack
        lidar_stack = get_stack(
            odom=self.data["odom"], lidar_img_dir=self.lidar_dir, i=index)

        if not self.include_lidar_file_names:
            lidar_stack = lidar_stack[0]

        # # get joystick vector and ten_meter_goal
        # joystick_data, ten_meter_goal = self.get_goal_and_joystick(index)
        joystick_data = self.data["future_joystick"][index]
        # pad with zeros if the length of the joystick sample is not long enough
        diff = max(0, self.joy_pred_len - joystick_data.shape[0])
        joystick_data = np.pad(joystick_data, pad_width=((0, diff), (0, 0)))
        joystick_data = joystick_data[: self.joy_pred_len, :]
        joystick_data = np.delete(joystick_data, 1, axis=1)

        # get 10m goal relative to the current position of the robot
        ten_meter_goal = self.data["local_goal_human_odom"][index][-1]
        ten_meter_goal = np.asarray(ten_meter_goal, dtype=np.float32)
        return lidar_stack, joystick_data, ten_meter_goal


class CLIPDataModule(pl.LightningDataModule):
    """Data wrapper for entire dataset

    setup training, validation splits
    """

    def __init__(self,
                 data_path: str,
                 batch_size: int,
                 num_workers=0,
                 ignore_first_n=30,
                 joy_len=300,
                 include_lidar_file_names=False,
                 pin_memory=False,
                 use_weighted_sampling=False,
                 verbose=False, ) -> None:
        """Initialize a pytorch lightning data module from a directory of processed rosbags

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
            pin_memory (bool, optional): toggle pin memory for pytorch dataloaders.
                Defaults to False.
            verbose (bool, optional): print datamodule information.
                Defaults to False.
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
        self.pin_memory = pin_memory
        self.use_weighted_sampling = use_weighted_sampling
        self.train_sampler_weights = []
        self.val_sampler_weights = []

        # diagnostics
        self.include_lidar_file_names = include_lidar_file_names
        self.verbose = verbose

        # load the pickle files
        self.pickle_file_paths = glob.glob(
            os.path.join(self.data_dir, "*_final.pkl"))
        self.pickle_file_paths = sorted(self.pickle_file_paths,
                                        key=lambda x: os.stat(x).st_size,
                                        reverse=True)

    def _generate_weight(self,
                         joystick_sample: np.ndarray,
                         goal_sample: np.ndarray, angular_weight=2.0) -> float:
        joystick_sample = np.abs(joystick_sample)
        lin_x_mu, ang_z_mu = np.mean(joystick_sample, axis=0)
        lin_x_sig, ang_z_sig = np.std(joystick_sample, axis=0)
        mu_weight = np.exp(2.0 - lin_x_mu) + np.exp(ang_z_mu * angular_weight)
        sig_weight = np.exp(lin_x_sig + ang_z_sig)
        goal_x, goal_y = goal_sample
        goal_x_weight = np.exp(12.0 - goal_x)
        goal_y_weight = np.exp(abs(goal_y) * angular_weight)
        return mu_weight + sig_weight + goal_x_weight + goal_y_weight

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup training and validation splits

        :param stage: one of either ['fit', 'validate', 'test', 'predict']
        """
        if stage in (None, "fit"):
            # print information
            cprint(
                f"loading data from {self.data_dir}...", "green", attrs=["bold"])
            if self.verbose:
                cprint(
                    f"skip first {self.ignore_first_n + STACK_LEN} frames", "cyan")
                cprint(f"batch size: {self.batch_size}", "cyan")
                cprint(f"future joystick length: {self.joy_len}\n", "cyan")

            # setup train/val split
            train_pkl = self.pickle_file_paths[: int(
                0.7 * len(self.pickle_file_paths))]
            val_pkl = self.pickle_file_paths[int(
                0.7 * len(self.pickle_file_paths)):]

            # load training pkl files
            if self.verbose:
                cprint("creating training set...", "green")
            for pfp in tqdm(train_pkl, position=0, leave=True):
                tmp_dataset = CLIPSet(pickle_file_path=pfp,
                                      ignore_first_n=self.ignore_first_n,
                                      future_joy_len=self.joy_len,
                                      include_lidar_file_names=self.include_lidar_file_names)

                # generate weights for each sample
                # in the training set
                if self.use_weighted_sampling:
                    for i in range(len(tmp_dataset)):
                        sample = tmp_dataset[i]
                        joystick = sample[1]
                        goal = sample[2]
                        weight = self._generate_weight(joystick, goal)
                        self.train_sampler_weights.append(weight)

                self.training_set.append(tmp_dataset)

            # load validation pkl files
            if self.verbose:
                cprint("creating validation set...", "green")
            for pfp in tqdm(val_pkl, position=0, leave=True):
                tmp_dataset = CLIPSet(pickle_file_path=pfp,
                                      ignore_first_n=self.ignore_first_n,
                                      future_joy_len=self.joy_len,
                                      include_lidar_file_names=self.include_lidar_file_names)
                # generate weights for each sample
                # in the validation set
                if self.use_weighted_sampling:
                    for i in range(len(tmp_dataset)):
                        sample = tmp_dataset[i]
                        joystick = sample[1]
                        goal = sample[2]
                        weight = self._generate_weight(joystick, goal)
                        self.val_sampler_weights.append(weight)

                self.validation_set.append(tmp_dataset)

            # concat dataset objects into full training and validation set
            self.training_set, self.validation_set = ConcatDataset(datasets=self.training_set), \
                ConcatDataset(
                datasets=self.validation_set)

            # ensure that training set is larger than validation set
            # for small bag counts
            if len(self.training_set) < len(self.validation_set):
                self.training_set, self.validation_set = (
                    self.validation_set, self.training_set)

            # display information
            if self.verbose:
                cprint(
                    f"training size: {len(self.training_set)} samples", "cyan")
                cprint(
                    f"validation size: {len(self.validation_set)} samples", "cyan")

    def train_dataloader(self) -> DataLoader:
        """return training dataloader

        Returns:
            DataLoader: training DataLoader
        """
        if self.use_weighted_sampling:
            sampler = WeightedRandomSampler(weights=self.train_sampler_weights,
                                            num_samples=len(
                                                self.train_sampler_weights),
                                            replacement=False)

            return DataLoader(self.training_set,
                              batch_size=self.batch_size,
                              sampler=sampler,
                              num_workers=self.num_workers,
                              pin_memory=self.pin_memory,
                              drop_last=True)

        return DataLoader(self.training_set,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=True)

    def val_dataloader(self) -> DataLoader:
        """return validation dataloader

        Returns:
            DataLoader: validation DataLoader
        """
        if self.use_weighted_sampling:
            sampler = WeightedRandomSampler(weights=self.val_sampler_weights,
                                            num_samples=len(
                                                self.val_sampler_weights),
                                            replacement=False)
            return DataLoader(self.validation_set,
                              batch_size=self.batch_size,
                              sampler=sampler,
                              num_workers=self.num_workers,
                              pin_memory=self.pin_memory,
                              drop_last=True)

        return DataLoader(self.validation_set,
                          shuffle=False,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=True)
