import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

### MATRIX TRANSFORMATIONS ###


def affineinverse(M):
    tmp = np.hstack((M[:2, :2].T, -M[:2, :2].T @ M[:2, 2].reshape((2, 1))))
    return np.vstack((tmp, np.array([0, 0, 1])))


def get_affine_matrix_quat(x, y, quaternion):
    theta = R.from_quat(quaternion).as_euler('XYZ')[2]
    return np.array([[np.cos(theta), -np.sin(theta), x],
                     [np.sin(theta), np.cos(theta), y], [0, 0, 1]])


### LIDAR DATA HANDLER ###


class BEVLidar:
    """ Class for handling lidar data and converting lidar points
        to a bird's-eye view image of the scene
    """

    def __init__(self,
                 x_range=(-20, 20),
                 y_range=(-20, 20),
                 z_range=(-1, 5),
                 resolution=0.05,
                 threshold_z_range=False):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.resolution = resolution
        self.dx = x_range[1] / resolution
        self.dy = y_range[1] / resolution
        self.img_size = int(1 + (x_range[1] - x_range[0]) / resolution)
        self.threshold_z_range = threshold_z_range

    def get_bev_lidar_img(self, lidar_points):
        img = np.zeros((self.img_size, self.img_size))
        for x, y, z in lidar_points:
            if self.not_in_range_check(x, y, z):
                continue
            ix = (self.dx + int(x / self.resolution))
            iy = (self.dy - int(y / self.resolution))
            if self.threshold_z_range:
                img[int(round(iy)),
                    int(round(ix))] = 1 if z >= self.z_range[0] else 0
            else:
                img[int(round(iy)), int(round(ix))] = (z - self.z_range[0]) / (
                    self.z_range[1] - self.z_range[0])
        return img

    def not_in_range_check(self, x, y, z):
        if x < self.x_range[0] \
            or x > self.x_range[1] \
            or y < self.y_range[0] \
            or y > self.y_range[1] \
            or z < self.z_range[0] \
                or z > self.z_range[1]:
            return True
        return False


def convert_float64img_to_uint8(image):
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    return image


def get_mask(img, visualize=False):
    mask_img = np.ones_like(img) * 255
    for i in range(int(mask_img.shape[0] / 2)):
        for j in range(i):
            mask_img[i, j] = 0
    for i in range(int(mask_img.shape[0] / 2), mask_img.shape[0]):
        for j in range(0, mask_img.shape[1] - i - 1):
            mask_img[i, j] = 0

    if visualize:
        cv2.imshow('disp', mask_img)
        cv2.waitKey(0)

    return mask_img