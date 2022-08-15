#!/usr/bin/env python
from argparse import ArgumentParser
from multiprocessing import Pool
import os
import pathlib
import pickle
import glob

import numpy as np
import sensor_msgs.point_cloud2 as pc2
from termcolor import cprint
import yaml
import cv2

from bev_lidar import *


def process_pkl(path_to_pkl, robot_config):
    with open(path_to_pkl, 'rb', 0) as p:
        data = pickle.load(p, encoding='latin1')
        length = len(data['lidar_msgs_sync'])
        cprint(f'total length of data: {length}\n', 'green', attrs=['bold'])

        # create BEVLidar handler
        bev_lidar_handler = BEVLidar(x_range=(-robot_config['LIDAR_RANGE_METERS'], robot_config['LIDAR_RANGE_METERS']),
                                     y_range=(-robot_config['LIDAR_RANGE_METERS'],
                                              robot_config['LIDAR_RANGE_METERS']),
                                     z_range=(-0.5,
                                              robot_config['LIDAR_HEIGHT_METERS']),
                                     resolution=robot_config['RESOLUTION'], threshold_z_range=False)

        lidar_save_path = os.path.join(path_to_pkl[:-4], 'lidar')
        pathlib.Path(lidar_save_path).mkdir(parents=True, exist_ok=True)
        cprint(
            f'saving lidar data to {lidar_save_path}', 'green', attrs=['bold'])
        # parse lidar messages and convert to bird's-eye view images
        for index, lidar in enumerate(data['lidar_msgs_sync']):
            lidar_points = pc2.read_points(
                lidar, skip_nans=True, field_names=("x", "y", "z"))
            lidar_img = bev_lidar_handler.get_bev_lidar_img(lidar_points)
            lidar_img = convert_float64img_to_uint8(lidar_img)

            # block out lidar image at the back of the robot
            img_mask = get_mask(lidar_img, visualize=False)
            lidar_img = cv2.bitwise_and(lidar_img, lidar_img, mask=img_mask)

            # save the
            file_path = os.path.join(lidar_save_path, f'{index}.png')


if __name__ == '__main__':
    # command line arguments
    parser = ArgumentParser(
        description='turn data from rosbags into data usable for the model')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(),
                        help='number of cores to use when processing')
    parser.add_argument('--path_to_data', type=str,
                        default='/home/abhinavchadaga/CS/clip_social_nav/data')
    parser.add_argument('--robot_config_path', type=str,
                        default='/home/abhinavchadaga/CS/clip_social_nav/robot_config/spot.yaml')
    args = parser.parse_args()

    # find all pickle files in the data dir
    pickle_file_paths = glob.glob(os.path.join(args.path_to_data, "*.pkl*"))
    cprint(f'found {len(pickle_file_paths)} pkl files to process:',
           'white', attrs=['bold'])
    for p in pickle_file_paths:
        print(p)
    print('\n')

    # load the robot configuration information
    robot_config = yaml.safe_load(open(args.robot_config_path))

    process_pkl(
        '/home/abhinavchadaga/CS/clip_social_nav/data/2021-11-12-14-28-49.pkl', robot_config)
