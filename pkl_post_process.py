import glob
import threading
import os
import pickle
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import directed_hausdorff
import scipy
import argparse
from tqdm import tqdm
from termcolor import cprint
from scipy.ndimage import maximum_filter1d
import random
from PIL import Image


def affineinverse(M):
    tmp = np.hstack((M[:2, :2].T, -M[:2, :2].T @ M[:2, 2].reshape((2, 1))))
    return np.vstack((tmp, np.array([0, 0, 1])))


def get_affine_matrix_quat(x, y, quaternion):
    theta = R.from_quat(quaternion).as_euler('XYZ')[2]
    return np.array([[np.cos(theta), -np.sin(theta), x],
                     [np.sin(theta), np.cos(theta), y], [0, 0, 1]])


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


def crop_image(img):
    return img[:400, :400]


def process_data(pickle_file_path, num_goal_pts):
    verbose = os.getenv('VERBOSE')

    print('processing pickle file : ', pickle_file_path)
    data = pickle.load(open(pickle_file_path, 'rb', 0), encoding='latin1')
    bev_lidar_dir = pickle_file_path[:-4]

    # create directory for the processed bev_lidar_imgs
    processed_bev_lidar_dir = bev_lidar_dir.replace('_data', '_final')
    if not os.path.exists(processed_bev_lidar_dir):
        os.makedirs(processed_bev_lidar_dir)

    # img mask to block out the lidar image at the back of the robot
    bev_lidar_img_0 = crop_image(
        np.array(Image.open(os.path.join(bev_lidar_dir, '1.png')),
                 dtype=np.uint8))
    img_mask = crop_image(get_mask(bev_lidar_img_0, visualize=False))

    # process each bev_lidar_img in the directory
    for file in os.listdir(bev_lidar_dir):
        img = np.array(Image.open(os.path.join(bev_lidar_dir, file)),
                       dtype=np.uint8)
        img = crop_image(img)
        img = cv2.bitwise_and(img, img, mask=img_mask)
        cv2.imwrite(os.path.join(processed_bev_lidar_dir, file), img)

    print('saved processed images to ', processed_bev_lidar_dir)

    data['local_goals'], data['local_goal_human_odom'] = [], []

    for i in range(len(os.listdir(bev_lidar_dir))):

        # get the next num_goal_pts local goals
        local_goal_list = []
        T_odom_robot = get_affine_matrix_quat(data['odom'][i][0],
                                              data['odom'][i][1],
                                              data['odom'][i][2])

        local_goal_human_odom = []

        if data['move_base_path'][i] is None:
            for goal in data['human_expert_odom'][i]:
                T_odom_goal = get_affine_matrix_quat(goal[0], goal[1], goal[2])
                T_robot_goal = affineinverse(T_odom_robot) @ T_odom_goal
                local_goal_list.append(
                    [T_robot_goal[0, 2], T_robot_goal[1, 2]])
        else:
            for goal in data['move_base_path'][i]:
                T_odom_goal = get_affine_matrix_quat(goal[0], goal[1], goal[2])
                T_robot_goal = affineinverse(T_odom_robot) @ T_odom_goal
                local_goal_list.append(
                    [T_robot_goal[0, 2], T_robot_goal[1, 2]])

        if data['human_expert_odom'] is None:
            raise Exception("no human expert path found!")

        if len(local_goal_list) > num_goal_pts:
            # more than 200 points exist on the blue line - need to subsample
            local_goal_list = [
                local_goal_list[i] for i in sorted(
                    random.sample(range(len(local_goal_list)), num_goal_pts))
            ]

        # add the human expert odom goals
        for goal in data['human_expert_odom'][i][:num_goal_pts]:
            T_odom_goal = get_affine_matrix_quat(goal[0], goal[1], goal[2])
            T_robot_goal = affineinverse(T_odom_robot) @ T_odom_goal
            local_goal_human_odom.append(
                [T_robot_goal[0, 2], T_robot_goal[1, 2]])

        if len(local_goal_list) < num_goal_pts:
            for _ in range(num_goal_pts - len(local_goal_list)):
                local_goal_list = local_goal_list + [local_goal_list[-1]]

        # if len(local_goal_human_odom) == 0 , we reached the end of the trajectory,
        # so we can skip this
        if len(local_goal_human_odom) == 0:
            continue

        if len(local_goal_human_odom) < num_goal_pts:
            for _ in range(num_goal_pts - len(local_goal_human_odom)):
                local_goal_human_odom = local_goal_human_odom + \
                    [local_goal_human_odom[-1]]

        data['local_goals'].append(np.asarray(local_goal_list))
        data['local_goal_human_odom'].append(np.asarray(local_goal_human_odom))

        # visualize these plans
        # img = cv2.cvtColor(data['bevlidarimg'][i], cv2.COLOR_GRAY2BGR)
        # for x in range(len(data['local_goals'][i])):
        # 	t_f_pixels = [int(data['local_goals'][i][x][0] / 0.05) + 200, int(-data['local_goals'][i][x][1] / 0.05) + 200]
        # 	img = cv2.circle(img, (t_f_pixels[0], t_f_pixels[1]), 1, (255, 0, 0), -1)
        # 	t_f_pixels = [int(data['local_goal_human_odom'][i][x][0] / 0.05) + 200,
        # 				  int(-data['local_goal_human_odom'][i][x][1] / 0.05) + 200]
        # 	img = cv2.circle(img, (t_f_pixels[0], t_f_pixels[1]), 1, (0, 0, 255), -1)
        #
        # cv2.imshow('disp2', img)
        # cv2.waitKey(0)

    data['odom_history'] = np.asarray(data['odom_history'])
    data['joystick'] = np.asarray(data['joystick'])
    data['joystick'] = np.delete(data['joystick'], 1, axis=1)
    # future joystick list will be composed of many
    # differently sized lists of joysticks values
    data['future_joystick'] = [
        np.asarray(j, dtype=np.float32) for j in data['future_joystick']
    ]

    data['local_goals'] = np.asarray(data['local_goals'])

    # cmd vel from move_base
    for i in range(len(data['move_base_cmd_vel'])):
        if data['move_base_cmd_vel'][i][0] is None:
            data['move_base_cmd_vel'][i] = [0.0, 0.0]
        else:
            data['move_base_cmd_vel'][i] = [
                data['move_base_cmd_vel'][i][0],
                data['move_base_cmd_vel'][i][2]
            ]

    data['move_base_cmd_vel'] = np.asarray(data['move_base_cmd_vel'])

    # now save the data in a _final.pkl file which is the processed data
    pickle.dump(
        data, open(pickle_file_path.replace('_data.pkl', '_final.pkl'), 'wb'))
    print('!!! Finished processing pickle file : ', pickle_file_path, ' !!!')


if __name__ == '__main__':
    # setup argparse
    parser = argparse.ArgumentParser(
        description='Process all the pickle files in parallel')
    parser.add_argument('--num_workers',
                        type=int,
                        default=1,
                        help='Number of independent threads to use')
    parser.add_argument('--data_path',
                        type=str,
                        default='/home/abhinavchadaga/CS/clip_social_nav/data')
    args = parser.parse_args()

    unprocessed_pkl_file_list = glob.glob(
        os.path.join(args.data_path, '*_data.pkl'))
    cprint('Found ' + str(len(unprocessed_pkl_file_list)) + ' pickle files...',
           'green',
           attrs=['bold'])
    unprocessed_pkl_file_list = [
        val for val in unprocessed_pkl_file_list
        if not os.path.exists(val.replace('_data.pkl', '_final.pkl'))
    ]
    cprint(str(len(unprocessed_pkl_file_list)) +
           ' pickle files need processing',
           'green',
           attrs=['bold'])

    num_workers = min(len(unprocessed_pkl_file_list), args.num_workers)
    cprint('Utilizing ' + str(num_workers) + ' workers for this job..',
           'white',
           attrs=['bold'])

    thread_list = []
    for unprocessed_pkl_file in tqdm(unprocessed_pkl_file_list):
        # create threads
        process_thread = threading.Thread(target=process_data,
                                          args=(unprocessed_pkl_file, 200))
        thread_list.append(process_thread)

        # if workers limit reached, start all the workers
        if len(thread_list) == num_workers:
            for i in range(num_workers):
                thread_list[i].start()

            # wait until all the workers have completed their jobs
            for i in range(num_workers):
                thread_list[i].join()

            thread_list = []

    cprint('Done processing all the pickle files !!', 'green', attrs=['bold'])
