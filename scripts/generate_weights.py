from multiprocessing import Pool
import os
from os.path import join, isdir
import glob
from argparse import ArgumentParser
import pathlib
import pickle

import numpy as np
from termcolor import cprint


def generate_weight(goal, joystick) -> float:
    joystick = np.abs(joystick)
    lin_x_mu, _, ang_z_mu = np.mean(joystick, axis=0)
    lin_x_sig, _, ang_z_sig = np.std(joystick, axis=0)
    mu_weight = (2.0 - lin_x_mu) + (ang_z_mu)
    sig_weight = lin_x_sig + ang_z_sig
    goal_x, goal_y = goal
    goal_x_weight = 12.0 - goal_x
    goal_y_weight = abs(goal_y)
    return mu_weight + sig_weight + goal_x_weight + goal_y_weight


def generate_weights(dir):
    # get joystick and goal directory
    joystick_dir = join(dir, 'joystick')
    goals_dir = join(dir, 'goals')

    # create save path for weights
    weights_dir = join(dir, 'weights')
    pathlib.Path(weights_dir).mkdir(parents=True, exist_ok=True)

    length = len(os.listdir(joystick_dir))
    weights = []
    for i in range(length):
        # load goal sample
        goal = join(goals_dir, f'{i}.pkl')
        goal = pickle.load(open(goal, 'rb'))

        # load joystick sample
        joystick = join(joystick_dir, f'{i}.pkl')
        joystick = pickle.load(open(joystick, 'rb'))

        # generate weight
        weight = generate_weight(goal, joystick)
        weights.append(weight)

    # dump all weights into one file
    pickle.dump(weights, open(join(weights_dir, 'weights.pkl'), 'wb'))


if __name__ == "__main__":
    # command line arguments
    parser = ArgumentParser(description='prepare datasets')
    parser.add_argument('--num_workers',
                        type=int,
                        default=os.cpu_count(),
                        help='number of cores to use when processing')
    parser.add_argument('--path_to_data',
                        type=str,
                        default='/home/abhinavchadaga/CS/clip_social_nav/data')
    args = parser.parse_args()

    path_to_data = args.path_to_data
    # get path to data, filter out pkl files
    data_dirs = glob.glob(join(path_to_data, "*"))
    data_dirs = [d for d in data_dirs if isdir(d)]

    cprint('generating weights for : ', 'white', attrs=['bold'])
    for d in data_dirs:
        print(d)

    num_workers = min(len(data_dirs), args.num_workers)
    with Pool(num_workers) as p:
        p.map(generate_weights, data_dirs)

    cprint('done generating weights', 'green', attrs=['bold'])