import math
import os
import warnings
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R
from skimage.measure import block_reduce
from torch import Tensor

# 10hz * 2 seconds = 20 frames
# used to get the previous two seconds
# of BEV Lidar data
STACK_LEN = 20


def affineinverse(M) -> np.ndarray:
    tmp = np.hstack((M[:2, :2].T, -M[:2, :2].T @ M[:2, 2].reshape((2, 1))))
    return np.vstack((tmp, np.array([0, 0, 1])))


def get_affine_matrix_quat(x, y, quaternion) -> np.ndarray:
    theta = R.from_quat(quaternion).as_euler('XYZ')[2]
    return np.array([[np.cos(theta), -np.sin(theta), x], [np.sin(theta), np.cos(theta), y],
                     [0, 0, 1]])


def get_stack(odom: list, lidar_img_dir: str, i: int) -> Tuple[np.ndarray, np.ndarray]:
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
        img = np.asarray(img, dtype=np.float32)
        lidar_stack.append(img)

    odom_stack = odom[i - STACK_LEN + 1:i + 1]
    lidar_stack = [lidar_stack[i] for i in [0, 5, 10, 15, 19]]
    odom_stack = [odom_stack[i] for i in [0, 5, 10, 15, 19]]

    # for visualization purposes
    img_file_names = [x for x in range(i - STACK_LEN + 1, i + 1)]
    img_file_names = [img_file_names[i] for i in [0, 5, 10, 15, 19]]

    # rotate previous frames to current frame
    last_frame = odom_stack[-1]
    t_odom_5 = get_affine_matrix_quat(last_frame[0], last_frame[1], last_frame[2])
    t_odom_4 = get_affine_matrix_quat(odom_stack[-2][0], odom_stack[-2][1], odom_stack[-2][2])
    t_4_5 = affineinverse(t_odom_4) @ t_odom_5
    t_odom_3 = get_affine_matrix_quat(odom_stack[-3][0], odom_stack[-3][1], odom_stack[-3][2])
    t_3_5 = affineinverse(t_odom_3) @ t_odom_5
    t_odom_2 = get_affine_matrix_quat(odom_stack[-4][0], odom_stack[-4][1], odom_stack[-4][2])
    t_2_5 = affineinverse(t_odom_2) @ t_odom_5
    t_odom_1 = get_affine_matrix_quat(odom_stack[-5][0], odom_stack[-5][1], odom_stack[-5][2])
    t_1_5 = affineinverse(t_odom_1) @ t_odom_5
    # now do the rotations
    t_1_5[:, -1] *= -20
    t_2_5[:, -1] *= -20
    t_3_5[:, -1] *= -20
    t_4_5[:, -1] *= -20
    lidar_stack[0] = cv2.warpAffine(lidar_stack[0], t_1_5[:2, :], (400, 400))
    lidar_stack[1] = cv2.warpAffine(lidar_stack[1], t_2_5[:2, :], (400, 400))
    lidar_stack[2] = cv2.warpAffine(lidar_stack[2], t_3_5[:2, :], (400, 400))
    lidar_stack[3] = cv2.warpAffine(lidar_stack[3], t_4_5[:2, :], (400, 400))
    lidar_stack = np.asarray(lidar_stack, dtype=np.float32)

    # mean and variance of height
    # 5, 400, 400 -> 10, 100, 100
    mv_stack = []
    for i in range(5):
        height_mean = block_reduce(lidar_stack[i],
                                   block_size=(4, 4),
                                   func=np.mean,
                                   func_kwargs={'dtype': np.float32})
        height_variance = block_reduce(lidar_stack[i],
                                       block_size=(4, 4),
                                       func=np.var,
                                       func_kwargs={'dtype': np.float32})
        mv_stack.extend([height_mean, height_variance])

    # combine the 5 single-channel images into a single image of 5 channels
    lidar_stack = np.asarray(mv_stack).astype(np.float32)
    lidar_stack = lidar_stack / 255.0  # normalize the image
    img_file_names = np.array(img_file_names)
    return lidar_stack, img_file_names


def visualize_lidar_stack(lidar_stack: np.ndarray, file_names: np.ndarray) -> None:
    file_names = [f'{n}.png' for n in file_names]
    rows, cols = 1, 5
    plt.figure(figsize=(25, 25))
    for i, (img, title) in enumerate(zip(lidar_stack, file_names)):
        plt.subplot(rows, cols, i + 1)
        plt.title(f'{title}, index: {i}')
        plt.imshow(img, cmap='gray')
    plt.show()


def visualize_goal(lidar_frame: np.ndarray, goal):
    # display the stuff
    lidar_frame = cv2.cvtColor(lidar_frame, cv2.COLOR_GRAY2BGR)
    t_f_pixels = [int(goal[0] / 0.05) + 200, int(-goal[1] / 0.05) + 200]
    lidar_frame = cv2.circle(lidar_frame, (t_f_pixels[0], t_f_pixels[1]), 5, (255, 0, 0), -1)

    return lidar_frame


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.", stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor
