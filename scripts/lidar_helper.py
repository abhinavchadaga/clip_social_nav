import os
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

# 10hz * 2 seconds = 20 frames
# used to get the previous two seconds
# of BEV Lidar data
STACK_LEN = 20


def affineinverse(M) -> np.ndarray:
    tmp = np.hstack((M[:2, :2].T, -M[:2, :2].T @ M[:2, 2].reshape((2, 1))))
    return np.vstack((tmp, np.array([0, 0, 1])))


def get_affine_matrix_quat(x, y, quaternion) -> np.ndarray:
    theta = R.from_quat(quaternion).as_euler('XYZ')[2]
    return np.array([[np.cos(theta), -np.sin(theta), x],
                     [np.sin(theta), np.cos(theta), y], [0, 0, 1]])


def get_stack(odom: list, lidar_img_dir: str,
              i: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Get a stack of 5 lidar images given index time = t

    :param odom: list of odometry values from the robot
    :param lidar_img_dir: path to the lidar image data
    :param i: index to get data from; corresponds to the file names of the lidar image data
    :return: a stack of lidar images, five channels, with current time t at index 4, and time t -
    2 at index 0
    """
    lidar_stack = []
    for x in range(i - STACK_LEN + 1, i + 1):
        img = Image.open(os.path.join(lidar_img_dir, f'{x}.png'))
        img = np.asarray(img)
        lidar_stack.append(img)

    odom_stack = odom[i - STACK_LEN + 1:i + 1]
    lidar_stack = [lidar_stack[i] for i in [0, 5, 10, 15, 19]]
    odom_stack = [odom_stack[i] for i in [0, 5, 10, 15, 19]]

    # for visualization purposes
    img_file_names = [x for x in range(i - STACK_LEN + 1, i + 1)]
    img_file_names = [img_file_names[i] for i in [0, 5, 10, 15, 19]]

    # rotate previous frames to current frame
    last_frame = odom_stack[-1]
    t_odom_5 = get_affine_matrix_quat(last_frame[0], last_frame[1],
                                      last_frame[2])
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
    lidar_stack[0] = cv2.warpAffine(lidar_stack[0], t_1_5[:2, :], (401, 401))
    lidar_stack[1] = cv2.warpAffine(lidar_stack[1], t_2_5[:2, :], (401, 401))
    lidar_stack[2] = cv2.warpAffine(lidar_stack[2], t_3_5[:2, :], (401, 401))
    lidar_stack[3] = cv2.warpAffine(lidar_stack[3], t_4_5[:2, :], (401, 401))

    # combine the 5 single-channel images into a single image of 5 channels
    lidar_stack = np.asarray(lidar_stack).astype(np.float32)
    lidar_stack = lidar_stack / 255.0  # normalize the image
    img_file_names = np.array(img_file_names)
    return lidar_stack, img_file_names


def visualize_lidar_stack(lidar_stack: np.ndarray,
                          file_names: np.ndarray) -> None:
    """

    :param lidar_stack:
    :param file_names:
    :return:
    """
    file_names = [f'{n}.png' for n in file_names]
    rows, cols = 1, 5
    plt.figure(figsize=(25, 25))
    for i, (img, title) in enumerate(zip(lidar_stack, file_names)):
        plt.subplot(rows, cols, i + 1)
        plt.title(f'{title}, index: {i}')
        plt.imshow(img, cmap='gray')
    plt.show()
