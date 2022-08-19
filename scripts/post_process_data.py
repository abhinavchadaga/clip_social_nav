#!/usr/bin/env python
from argparse import ArgumentParser
import itertools
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

from utils import *


def lidar_msg_to_img(lidar_msg, bev_lidar_handler) -> np.ndarray:
    """ Convert a lidar message into bird's eye view image

    Args:
        lidar_msg: message to convert
        bev_lidar_handler (BEVLidar): configuration to convert msg

    Return:
        np.ndarray: bev lidar image
    """
    # convert message to points
    lidar_points = pc2.read_points(lidar_msg,
                                   skip_nans=True,
                                   field_names=("x", "y", "z"))

    # convert points to img
    lidar_img = bev_lidar_handler.get_bev_lidar_img(lidar_points)
    lidar_img = convert_float64img_to_uint8(lidar_img)

    # block out lidar image at the back of the robot
    img_mask = get_mask(lidar_img, visualize=False)
    lidar_img = cv2.bitwise_and(lidar_img, lidar_img, mask=img_mask)
    return lidar_img


def crop_image(img: np.ndarray) -> np.ndarray:
    """ Crop lidar image from 241 x 241 to 240 x 240

    Args:
        img (np.ndarray): lidar image to crop
    """
    return img[:240, :240]


def get_goal_odom(odom_sync_msg, odom_msgs, odom_ts, current_time):
    closest_index = np.searchsorted(odom_ts, current_time) + 1
    goal_index = min(closest_index + 30, len(odom_msgs) - 1)
    goal_odom = odom_msgs[goal_index]
    for goal_index in range(closest_index, len(odom_msgs)):
        goal_odom = odom_msgs[goal_index]
        dist = np.linalg.norm(
            np.array([odom_sync_msg[0], odom_sync_msg[1]]) -
            np.array([goal_odom[0], goal_odom[1]]))
        if dist > 6.0:
            break

    traj = odom_msgs[closest_index:goal_index][:160]
    traj_ts = odom_ts[closest_index:goal_index][:160]
    goal_odom = traj[-1]
    goal_ts = traj_ts[-1]
    return goal_odom, goal_ts


def transform_goal(current_odom, goal_odom):
    t_odom_robot = get_affine_matrix_quat(current_odom[0], current_odom[1],
                                          current_odom[2])
    t_odom_goal = get_affine_matrix_quat(goal_odom[0], goal_odom[1],
                                         goal_odom[2])
    t_robot_goal = affineinverse(t_odom_robot) @ t_odom_goal
    goal_odom = (t_robot_goal[0, 2], t_robot_goal[1, 2])
    return goal_odom


def get_joystick_values(current_time, joy_msgs, joy_ts, goal_ts):
    closest_index = np.searchsorted(joy_ts, current_time) + 1
    goal_index = np.searchsorted(joy_ts, goal_ts)
    future_joystick_data = joy_msgs[closest_index:goal_index]
    future_joystick_data = np.asarray(future_joystick_data, dtype=np.float32)
    return future_joystick_data


def process_pkl(path_to_pkl, robot_config):
    with open(path_to_pkl, 'rb', 0) as p:
        data = pickle.load(p, encoding='latin1')
        length = len(data['lidar_msgs_sync'])

        # create BEVLidar handler
        bev_lidar_handler = BEVLidar(
            x_range=(-robot_config['LIDAR_RANGE_METERS'],
                     robot_config['LIDAR_RANGE_METERS']),
            y_range=(-robot_config['LIDAR_RANGE_METERS'],
                     robot_config['LIDAR_RANGE_METERS']),
            z_range=(-0.5, robot_config['LIDAR_HEIGHT_METERS']),
            resolution=robot_config['RESOLUTION'],
            threshold_z_range=False)
        # define save paths
        base_dir = path_to_pkl[:-4]
        lidar_dir = os.path.join(base_dir, 'lidar_imgs')
        goal_dir = os.path.join(base_dir, 'goals')
        joystick_dir = os.path.join(base_dir, 'joystick')
        odom_dir = os.path.join(base_dir, 'odom')

        # create save directories
        pathlib.Path(lidar_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(goal_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(joystick_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(odom_dir).mkdir(parents=True, exist_ok=True)

        for i in range(length):
            # get current timestamp
            lidar_msg = data['lidar_msgs_sync'][i]
            current_time = lidar_msg.header.stamp.to_sec()

            # process lidar data
            lidar_img = lidar_msg_to_img(lidar_msg, bev_lidar_handler)
            lidar_img = crop_image(lidar_img)

            # save lidar img
            lidar_file_path = os.path.join(lidar_dir, f'{i}.png')
            cv2.imwrite(lidar_file_path, lidar_img)

            # save all odom data for lidar stack creation
            odom = data['odom_msgs_sync'][i]
            odom_file_path = os.path.join(odom_dir, f'{i}.pkl')
            pickle.dump(odom, open(odom_file_path, 'wb'))

            # get 6m goal and timestamp
            goal_odom, goal_ts = get_goal_odom(odom, data['odom_msgs'],
                                               data['odom_ts'], current_time)

            # make goal relative to current position of the robot
            goal_odom = np.asarray(transform_goal(current_odom=odom,
                                                  goal_odom=goal_odom),
                                   dtype=np.float32)

            # save goal data
            goal_file_path = os.path.join(goal_dir, f'{i}.pkl')
            pickle.dump(goal_odom, open(goal_file_path, 'wb'))

            # use goal timestamp to get joystick messages until the goal
            joystick_values = get_joystick_values(current_time,
                                                  data['joystick_msgs'],
                                                  data['joystick_ts'], goal_ts)

            # save joystick data
            joystick_file_path = os.path.join(joystick_dir, f'{i}.pkl')
            pickle.dump(joystick_values, open(joystick_file_path, 'wb'))

        cprint(f'finished processing {path_to_pkl}', 'green', attrs=['bold'])


if __name__ == '__main__':
    # command line arguments
    parser = ArgumentParser(
        description='turn data from rosbags into data usable for the model')
    parser.add_argument('--num_workers',
                        type=int,
                        default=os.cpu_count(),
                        help='number of cores to use when processing')
    parser.add_argument('--path_to_data',
                        type=str,
                        default='/home/abhinavchadaga/CS/clip_social_nav/data')
    parser.add_argument(
        '--robot_config_path',
        type=str,
        default='/home/abhinavchadaga/CS/clip_social_nav/robot_config/spot.yaml'
    )
    args = parser.parse_args()

    # find all pickle files in the data dir
    pickle_file_paths = glob.glob(os.path.join(args.path_to_data, "*.pkl*"))
    cprint(f'found {len(pickle_file_paths)} pkl files to process:',
           'white',
           attrs=['bold'])
    for p in pickle_file_paths:
        print(p)

    # load the robot configuration information
    robot_config = yaml.safe_load(open(args.robot_config_path))
    cprint('loaded robot config\n', 'white', attrs=['bold'])

    # process each pickle file on its own core
    num_workers = min(len(pickle_file_paths), args.num_workers)
    cprint(f'using {num_workers} workers\n', 'white', attrs=['bold'])
    with Pool(num_workers) as p:
        p.starmap(process_pkl,
                  zip(pickle_file_paths, itertools.repeat(robot_config)))
    cprint('post processing complete!', 'green', attrs=['bold'])
