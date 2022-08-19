from os import listdir, remove, rename, cpu_count
from os.path import join, isdir
from multiprocessing import Pool
import glob
from shutil import rmtree
import pathlib
import pickle
from argparse import ArgumentParser

import numpy as np
import cv2
from PIL import Image
from termcolor import cprint

from utils import affineinverse, get_affine_matrix_quat

STACK_LEN = 20


def create_lidar_stack(lidar_dir, odom_dir, index):
    # get past 20 lidar images into a list
    lidar_stack = []
    for i in range(index - STACK_LEN + 1, index + 1):
        img = Image.open(join(lidar_dir, f'{i}.png'))
        img = np.asarray(img, dtype=np.float32)
        lidar_stack.append(img)

    # get past 20 odom data points
    odom_stack = [
        pickle.load(open(join(odom_dir, f'{i}.pkl'), 'rb'))
        for i in range(index - STACK_LEN + 1, index + 1)
    ]

    # filter stacks to 5 data points each
    lidar_stack = [lidar_stack[i] for i in [0, 5, 10, 15, 19]]
    odom_stack = [odom_stack[i] for i in [0, 5, 10, 15, 19]]

    # rotate previous frames to current frame
    last_odom_frame = odom_stack[-1]
    t_odom_5 = get_affine_matrix_quat(last_odom_frame[0], last_odom_frame[1],
                                      last_odom_frame[2])
    t_odom_4 = get_affine_matrix_quat(odom_stack[-2][0], odom_stack[-2][1],
                                      odom_stack[-2][2])
    t_4_5 = affineinverse(t_odom_4) @ t_odom_5
    t_odom_3 = get_affine_matrix_quat(odom_stack[-3][0], odom_stack[-3][1],
                                      odom_stack[-3][2])
    t_3_5 = affineinverse(t_odom_3) @ t_odom_5
    t_odom_2 = get_affine_matrix_quat(odom_stack[-4][0], odom_stack[-4][1],
                                      odom_stack[-4][2])
    t_2_5 = affineinverse(t_odom_2) @ t_odom_5
    t_odom_1 = get_affine_matrix_quat(odom_stack[-5][0], odom_stack[-5][1],
                                      odom_stack[-5][2])
    t_1_5 = affineinverse(t_odom_1) @ t_odom_5
    # now do the rotations
    t_1_5[:, -1] *= -20
    t_2_5[:, -1] *= -20
    t_3_5[:, -1] *= -20
    t_4_5[:, -1] *= -20
    lidar_stack[0] = cv2.warpAffine(lidar_stack[0], t_1_5[:2, :], (240, 240))
    lidar_stack[1] = cv2.warpAffine(lidar_stack[1], t_2_5[:2, :], (240, 240))
    lidar_stack[3] = cv2.warpAffine(lidar_stack[3], t_4_5[:2, :], (240, 240))
    lidar_stack[2] = cv2.warpAffine(lidar_stack[2], t_3_5[:2, :], (240, 240))
    lidar_stack = np.asarray(lidar_stack, dtype=np.float32)
    return lidar_stack


def prepare_dataset(dir):
    # get paths to sub-dirs
    lidar_dir = join(dir, 'lidar_imgs')
    odom_dir = join(dir, 'odom')
    save_path = join(dir, 'lidar')

    # generate directory for new data
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

    length = len(listdir(lidar_dir))
    for i in range(STACK_LEN + skip_first_n, length):
        lidar_stack = create_lidar_stack(lidar_dir, odom_dir, i)
        file_path = join(save_path,
                         f'{i - STACK_LEN - skip_first_n}.pkl')
        pickle.dump(lidar_stack, open(file_path, 'wb'))

    # delete skipped first STACK_LEN + skip_first_n
    # goals and joystick data, rename rest
    goals_dir = join(dir, 'goals')
    joystick_dir = join(dir, 'joystick')
    for i in range(length):
        goal_file = join(goals_dir, f'{i}.pkl')
        joystick_file = join(joystick_dir, f'{i}.pkl')
        if i < STACK_LEN + skip_first_n:
            # delete
            remove(goal_file)
            remove(joystick_file)
        else:
            # rename remaining goal files
            new_goal_file_path = join(
                goals_dir, f'{i - STACK_LEN - skip_first_n}.pkl')
            rename(goal_file, new_goal_file_path)

            # rename remaining joystick files
            new_joystick_file_path = join(
                joystick_dir, f'{i - STACK_LEN - skip_first_n}.pkl')
            rename(joystick_file, new_joystick_file_path)

    # remove old lidar directory and odom dir
    # no longer needed
    rmtree(lidar_dir)
    rmtree(odom_dir)

    cprint(f'complete: {dir}', 'green', attrs=['bold'])


if __name__ == "__main__":
    # command line arguments
    parser = ArgumentParser(description='prepare datasets')
    parser.add_argument('--num_workers',
                        type=int,
                        default=cpu_count(),
                        help='number of cores to use when processing')
    parser.add_argument('--path_to_data',
                        type=str,
                        default='/home/abhinavchadaga/CS/clip_social_nav/data')
    parser.add_argument('--skip_first_n', type=int, default=30)
    args = parser.parse_args()

    skip_first_n = args.skip_first_n
    path_to_data = args.path_to_data
    # get path to data, filter out pkl files
    data_dirs = glob.glob(join(path_to_data, "*"))
    data_dirs = [d for d in data_dirs if isdir(d)]

    cprint('preparing dataset using: ', 'white', attrs=['bold'])
    for d in data_dirs:
        print(d)

    num_workers = min(len(data_dirs), args.num_workers)
    with Pool(num_workers) as p:
        p.map(prepare_dataset, data_dirs)

    cprint('finished preparing dataset', 'green', attrs=['bold'])
