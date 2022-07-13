import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from PIL import Image


def affineinverse(M) -> np.ndarray:
    tmp = np.hstack((M[:2, :2].T, -M[:2, :2].T @ M[:2, 2].reshape((2, 1))))
    return np.vstack((tmp, np.array([0, 0, 1])))


def get_affine_matrix_quat(x, y, quaternion) -> np.ndarray:
    theta = R.from_quat(quaternion).as_euler('XYZ')[2]
    return np.array([[np.cos(theta), -np.sin(theta), x],
                     [np.sin(theta), np.cos(theta), y],
                     [0, 0, 1]])


def get_bev_lidar_img_stack(odom: dict, path_to_bev_lidar: str, i: int, verbose=False):
    bev_img_stack = [np.array(Image.open(os.path.join(
        path_to_bev_lidar, f'{x}.png'))) for x in range(i - 20 + 1, i + 1)]
    bev_img_stack_odoms = odom[i - 20 + 1: i + 1]
    bev_img_stack = [bev_img_stack[i] for i in [0, 5, 10, 15, 19]]
    bev_img_stack_odoms = [bev_img_stack_odoms[i]
                           for i in [0, 5, 10, 15, 19]]

    # for visualization purposes
    if verbose:
        img_file_names = [f'{x}.png' for x in range(i - 20 + 1, i + 1)]
        img_file_names = [img_file_names[i]
                          for i in [0, 5, 10, 15, 19]]

    # rotate previous frames to current frame
    last_frame = bev_img_stack_odoms[-1]
    T_odom_5 = get_affine_matrix_quat(
        last_frame[0], last_frame[1], last_frame[2])
    T_odom_4 = get_affine_matrix_quat(bev_img_stack_odoms[-2][0],
                                      bev_img_stack_odoms[-2][1],
                                      bev_img_stack_odoms[-2][2])
    T_4_5 = affineinverse(T_odom_4) @ T_odom_5
    T_odom_3 = get_affine_matrix_quat(bev_img_stack_odoms[-3][0],
                                      bev_img_stack_odoms[-3][1],
                                      bev_img_stack_odoms[-3][2])
    T_3_5 = affineinverse(T_odom_3) @ T_odom_5
    T_odom_2 = get_affine_matrix_quat(bev_img_stack_odoms[-4][0],
                                      bev_img_stack_odoms[-4][1],
                                      bev_img_stack_odoms[-4][2])
    T_2_5 = affineinverse(T_odom_2) @ T_odom_5
    T_odom_1 = get_affine_matrix_quat(bev_img_stack_odoms[-5][0],
                                      bev_img_stack_odoms[-5][1],
                                      bev_img_stack_odoms[-5][2])
    T_1_5 = affineinverse(T_odom_1) @ T_odom_5
    # now do the rotations
    T_1_5[:, -1] *= -20
    T_2_5[:, -1] *= -20
    T_3_5[:, -1] *= -20
    T_4_5[:, -1] *= -20
    bev_img_stack[0] = cv2.warpAffine(
        bev_img_stack[0], T_1_5[:2, :], (401, 401))
    bev_img_stack[1] = cv2.warpAffine(
        bev_img_stack[1], T_2_5[:2, :], (401, 401))
    bev_img_stack[2] = cv2.warpAffine(
        bev_img_stack[2], T_3_5[:2, :], (401, 401))
    bev_img_stack[3] = cv2.warpAffine(
        bev_img_stack[3], T_4_5[:2, :], (401, 401))

    # combine the 5 single-channel images into a single image of 5 channels
    bev_img_stack = np.asarray(bev_img_stack).astype(np.float32)
    bev_img_stack = bev_img_stack / 255.0  # normalize the image
    if not verbose:
        return bev_img_stack,
    else:
        return bev_img_stack, img_file_names
