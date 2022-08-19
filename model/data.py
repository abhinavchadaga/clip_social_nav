from os import listdir
from os.path import join
import pickle
from typing import Tuple

from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler, DataLoader
import numpy as np
from termcolor import cprint


class CLIPSet(Dataset):
    """ Dataset object representing the data from a single rosbag
    """

    def __init__(self, data_dir: str, joy_len=300) -> None:
        """ initialize a CLIPSet object,
            save path to data but do not load into RAM

        Args:
            data_dir (str): path to the data pulled from a single rosbag
            future_joy_len (str): number of messages to use from each joystick data point
        """
        super().__init__()
        # save paths to lidar, joystick, goals and weights data
        self.lidar_dir = join(data_dir, 'lidar')
        self.joystick_dir = join(data_dir, 'joystick')
        self.goals_dir = join(data_dir, 'goals')
        self.weights_path = join(data_dir, 'weights/weights.pkl')
        # sanity check
        assert len(listdir(self.lidar_dir)) == len(listdir(
            self.joystick_dir)) == len(listdir(self.goals_dir))
        # save sequence length and future_joy_length
        self.length = len(listdir(self.lidar_dir))
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
        joystick = np.pad(joystick, pad_width=((0, diff), (0, 0)))
        joystick = joystick[:self.future_joy_len, :]
        joystick = np.delete(joystick, 1, axis=1)

        return lidar_stack, goal, joystick

    def load_weights(self):
        weights = pickle.load(open(self.weights_path, 'rb'))
        return weights


def create_dataloader(data_dirs,
                      batch_size,
                      num_workers,
                      pin_memory,
                      use_weighted_sampling,
                      train,
                      joy_len=300):
    tmp_sets = []
    weights = []
    for d in data_dirs:
        tmp = CLIPSet(d, joy_len)
        tmp_sets.append(tmp)
        if use_weighted_sampling:
            tmp_weights = tmp.load_weights()
            weights.extend(tmp_weights)

    # concatenate all datasets into one set
    dataset = ConcatDataset(tmp_sets)

    # if not using weighted sampling
    # set shuffle to true for training dataloader
    # and false to validation dataloader
    shuffle = True if train else False
    if use_weighted_sampling:
        sampler = WeightedRandomSampler(weights=weights,
                                        num_samples=len(weights),
                                        replacement=False)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=pin_memory,
                                drop_last=True,
                                sampler=sampler)
        return dataloader, len(dataset)
    else:
        dataloader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                pin_memory=pin_memory,
                                drop_last=True,
                                shuffle=shuffle)

        return dataloader, len(dataset)


if __name__ == "__main__":
    trainloader = create_dataloader(
        data_dirs='/home/abhinavchadaga/CS/clip_social_nav/data',
        batch_size=128,
        num_workers=8,
        pin_memory=True,
        use_weighted_sampling=True,
        joy_len=300)

    print(len(trainloader))

    for i, (lidar, goal, joystick) in enumerate(trainloader):
        if i == 0:
            print(lidar.shape)
            print(goal.shape)
            print(joystick.shape)

    cprint('success', 'green', attrs=['bold'])
