#!usr/bin/env python
import argparse
import os

import numpy as np
import pickle
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from sensor_msgs.msg import PointCloud2, Joy
import rospy
import message_filters
import os
import rosbag
import yaml
import math
import subprocess
from scipy.spatial.transform import Rotation as R
from termcolor import cprint
from tqdm.auto import tqdm

data = {'lidar': [1, 2, 3], 'odom': [1, 2, 3],
        'joystick': [1, 2, 3], 'goals': [1, 2, 3]}


def callback(lidar_msg, odom_msg, joystick_msg):
    pass


def save_data(save_path, rosbag_path):
    rosbag_name = rosbag_path.split('/')[-1]
    save_path = os.path.join(save_path, rosbag_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for k in data.keys():
        pickle.dump(data[k], open(os.path.join(save_path, k)))


def main():
    rospy.init_node('process_rosbag', anonymous=True)
    rosbag_path = rospy.get_param('rosbag_path')
    save_data_path = rospy.get_param('save_data_path')
    robot_name = rospy.get_param('robot_name')
    # check if path to rosbag exists
    if not os.path.exists(rosbag_path):
        raise Exception('invalid rosbag path')

    # setup subscribers
    lidar_sub = message_filters.Subscriber('/velodyne_points', PointCloud2)
    odom_sub = message_filters.Subscriber('/odom', Odometry)
    joystick_sub = message_filters.Subscriber('/joystick', Joy)

    # setup timesynchronizer
    time_synchronizer = message_filters.ApproximateTimeSynchronizer(
        [lidar_sub, odom_sub, joystick_sub], 100, 0.5, allow_headerless=True)
    time_synchronizer.registerCallback(callback)

    # load config for robot
    # find root of the ros node and config file path
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_file_path = os.path.join(
        package_root, 'robot_config/'+str(robot_name)+'.yaml')
    config = yaml.safe_load(config_file_path)

    # load rosbag
    rosbag = rosbag.Bag(rosbag_path)

    # pre-record all odom messages
    # these are not time synchronized with the lidar frames
    cprint('reading all odom messages...', 'green')
    odom_msgs = {}
    for topic, msg, t in tqdm(rosbag.read_messages(topics=['/odom'])):
        if len(odom_msgs) == 0:
            odom_msgs[0.0] = msg
        else:
            odom_msgs[t.to_sec()] = msg
    cprint('Done reading odom messages from the rosbag',
           color='green')

    # pre-record all the joystick messages
    # also not time synchronized with the lidar frames
    cprint(
        'Now reading all the joystick messages and timestamps from the rosbag',
        'green')
    joystick_msgs = {}
    for _, msg, t in tqdm(rosbag.read_messages(topics=['/joystick'])):
        if len(joystick_msgs) == 0:
            joystick_msgs[0.0] = msg
        else:
            joystick_msgs[t.to_sec()] = msg
    cprint('Done reading joystick messages and timestamps',
           color='green',
           attrs=['bold'])

    info_dict = yaml.safe_load(rosbag._get_yaml_info())
    duration = info_dict['end'] - info_dict['start']
    cprint('rosbag_length: ', duration)

    skip_end_sec = 6
    skip_start_sec = 10
    play_duration = str(
        int(math.floor(duration) - skip_end_sec - skip_start_sec))
    print('play duration: {}'.format(play_duration))

    rosbag_play_process = subprocess.Popen([
        'rosbag', 'play', rosbag_path, '-r', '1.0', '--clock', '-u',
        play_duration, '-s',
        str(skip_start_sec)
    ])

    while not rospy.is_shutdown():
        # check if the python process is still running
        if rosbag_play_process.poll() is not None:
            print('rosbag process has stopped')
            print('Data was saved in :: ', save_data_path)
            exit(0)

    rospy.spin()


if __name__ == "__main__":
    main()
