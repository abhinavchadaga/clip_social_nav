#!/usr/bin/env python
import os
import math
import pathlib
import subprocess
import pickle

import numpy as np
import rospy
import rosbag
import message_filters
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, Joy
import yaml
from termcolor import cprint
from tqdm.auto import tqdm

# data to save from rosbag
data = {'lidar_msgs_sync': [], 'odom_msgs_sync': [], 'joystick_msgs_sync': []}

### JOYSTICK DATA UTILS ###


def joystickValue(x, scale, kDeadZone=0.02) -> float:
    if kDeadZone != 0.0 and abs(x) < kDeadZone:
        return 0.0
    return ((x - np.sign(x) * kDeadZone) / (1.0 - kDeadZone) * scale)


def convert_joystick_msg(joystick_msg, robot_config) -> list:
    joy_axes = joystick_msg.axes
    linear_x = joystickValue(joy_axes[robot_config['kXAxis']],
                             -robot_config['kMaxLinearSpeed'])
    linear_y = joystickValue(joy_axes[robot_config['kYAxis']],
                             -robot_config['kMaxLinearSpeed'])
    angular_z = joystickValue(joy_axes[robot_config['kRAxis']],
                              -np.deg2rad(90.0),
                              kDeadZone=0.0)

    return [linear_x, linear_y, angular_z]


### ODOM DATA UTILS ###


def convert_odom_msg(odom_msg):
    x = odom_msg.pose.pose.position.x
    y = odom_msg.pose.pose.position.y
    orientation = odom_msg.pose.pose.orientation
    quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
    return [x, y, quaternion]


def callback(lidar, odom, joystick):
    # append synced lidar points
    data['lidar_msgs_sync'].append(lidar)

    odom = convert_odom_msg(odom)
    # save odom msg
    data['odom_msgs_sync'].append(odom)

    # save joystick msg
    data['joystick_msgs_sync'].append(joystick)


def main():
    rospy.init_node('rosbag_parser', anonymous=True)
    rosbag_path = rospy.get_param('rosbag_path')
    save_data_path = rospy.get_param('save_data_path')
    robot_config_path = rospy.get_param('robot_config_path')
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

    # load rosbag
    bag = rosbag.Bag(rosbag_path)
    # load robot config yaml
    robot_config_path = yaml.safe_load(open(robot_config_path))
    cprint('loaded robot config: ', 'white', attrs=['bold'])
    print(robot_config_path)

    # create data directory
    pathlib.Path(save_data_path).mkdir(parents=True, exist_ok=True)
    # use rosbag name to create path to pkl output file
    path_to_pkl = os.path.join(
        save_data_path,
        rosbag_path.split('/')[-1].replace('.bag', '.pkl'))

    # pre-record all odom messages
    # these are not time synchronized with the lidar frames
    # print('\n')
    odom_msgs, odom_ts = [], []
    cprint('Reading all odom messages...', 'green', attrs=['bold'])
    for _, msg, t in tqdm(bag.read_messages(topics=['/odom'])):
        msg = convert_odom_msg(msg)
        odom_msgs.append(msg)
        if len(odom_ts) == 0:
            odom_ts.append(0.0)
        else:
            odom_ts.append(t.to_sec())

    data['odom_msgs'] = odom_msgs
    data['odom_ts'] = odom_ts

    cprint('Done reading odom messages from the rosbag\n',
           color='green',
           attrs=['bold'])

    # pre-record all the joystick messages
    # also not time synchronized with the lidar frames
    cprint(
        'Now reading all the joystick messages and timestamps from the rosbag',
        'green',
        attrs=['bold'])
    joystick_msgs, joystick_ts = [], []
    for _, msg, t in tqdm(bag.read_messages(topics=['/joystick'])):
        joystick_msgs.append(convert_joystick_msg(msg, robot_config_path))
        if len(joystick_ts) == 0:
            joystick_ts.append(0.0)
        else:
            joystick_ts.append(t.to_sec())

    data['joystick_msgs'] = joystick_msgs
    data['joystick_ts'] = joystick_ts

    cprint('Done reading joystick messages and timestamps\n',
           color='green',
           attrs=['bold'])

    # load play information from rosbag
    info_dict = yaml.safe_load(bag._get_yaml_info())
    duration = info_dict['end'] - info_dict['start']
    cprint(f'rosbag_length: {duration:.2f} s', 'green', attrs=['bold'])

    # skip last six seconds and first ten seconds
    skip_end_sec = 6
    skip_start_sec = 10
    play_duration = str(
        int(math.floor(duration) - skip_end_sec - skip_start_sec))
    cprint(f'play duration: {play_duration} s', 'green', attrs=['bold'])

    rosbag_play_process = subprocess.Popen([
        'rosbag', 'play', rosbag_path, '-r', '1.0', '--clock', '-u',
        play_duration, '-s',
        str(skip_start_sec)
    ])

    while not rospy.is_shutdown():
        # check if the python process is still running
        if rosbag_play_process.poll() is not None:
            cprint('rosbag process has stopped', 'green', attrs=['bold'])
            pickle.dump(data, open(path_to_pkl, 'wb'))
            cprint(f'Data was saved to :: {path_to_pkl}',
                   'green',
                   attrs=['bold'])
            exit(0)

    rospy.spin()


if __name__ == "__main__":
    main()
