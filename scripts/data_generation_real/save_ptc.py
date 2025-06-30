import os
import tqdm
import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
import torchvision
from termcolor import cprint
import time

import glob
from PIL import Image
import pickle as pkl
from scipy.spatial.transform import Rotation as R

import cv2
from pdb import set_trace
import threading

def xyzypr2xyzquat(pose):
    x, y, z, yaw, pitch, roll = pose
    r = R.from_euler('ZYX', [yaw, pitch, roll])
    qx, qy, qz, qw = r.as_quat()
    return [x, y, z, qw, qx, qy, qz]

def random_point_sampling(points, num_points=1024, use_cuda=True):
    num_total_points = points.shape[0]
    if use_cuda:
        points = torch.from_numpy(points).cuda()
        indices = torch.randperm(num_total_points, device=points.device)[:num_points]
        sampled_points = points[indices]
        sampled_points = sampled_points.cpu().numpy()
        indices = indices.cpu().numpy()
    else:
        indices = np.random.choice(num_total_points, num_points, replace=False)
        sampled_points = points[indices]

    return sampled_points, indices

def farthest_point_sampling(points, num_points=1024, use_cuda=True):
    K = [num_points]
    if use_cuda:
        points = torch.from_numpy(points).cuda()
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.cpu().numpy()
    else:
        points = torch.from_numpy(points)
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.numpy()

    return sampled_points, indices

def preprocess_point_cloud(points, num_points=1024, use_cuda=True):
    
    # TODO: modify your workspace
    WORK_SPACE = [
        [-0.1, 1.2], # x
        [-1, 1], # y
        [0.02, 1] # z
    ]
    
     # crop
    origin_points = points[np.where((points[..., 0] > WORK_SPACE[0][0]) & (points[..., 0] < WORK_SPACE[0][1]) &
                                (points[..., 1] > WORK_SPACE[1][0]) & (points[..., 1] < WORK_SPACE[1][1]) &
                                (points[..., 2] > WORK_SPACE[2][0]) & (points[..., 2] < WORK_SPACE[2][1]))]

    points, sample_indices = random_point_sampling(origin_points, num_points, use_cuda=False)

    return points
   
def preproces_image(image, img_size_H, img_size_W, mode):

    image = image.astype(np.float32)
    image = torch.from_numpy(image).cuda()
    image = image.permute(2, 0, 1) # HxWx4 -> 4xHxW

    if mode == 'nearest':
        interpolation = torchvision.transforms.InterpolationMode.NEAREST
    elif mode == 'bicubic':
        interpolation = torchvision.transforms.InterpolationMode.BICUBIC

    image = torchvision.transforms.functional.resize(image, (img_size_H, img_size_W), interpolation)
    image = image.permute(1, 2, 0) # 4xHxW -> HxWx4
    image = image.cpu().numpy()
    return image

def get_pointcloud_from_multicameras(cameras, images, depths, extrs, intrs, H, W):
    multiview_pointcloud = None
    scale = H / 480
    for camera in cameras:
        depth_array = depths[camera]
        rgb_array = images[camera]
        intr = intrs[camera] * scale
        extr = extrs[camera]

        h, w = depth_array.shape
        v, u = np.indices((h, w))
        z = depth_array
        x = (u - intr[0, 2]) * z / intr[0, 0]
        y = (v - intr[1, 2]) * z / intr[1, 1]
        points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        
        # Apply the extrinsic transformation
        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        point_cloud = (extr @ points_homogeneous.T).T[:, :3] # N * 3
        rgb_point_cloud = np.concatenate((point_cloud, rgb_array.reshape(-1, 3)), axis=-1)

        if multiview_pointcloud is None:
            multiview_pointcloud = rgb_point_cloud
        else:
            multiview_pointcloud = np.concatenate((multiview_pointcloud, rgb_point_cloud), axis=0)

    return multiview_pointcloud

def process_multi_steps(steps_dir, start, end, task_name, save_data_path, pcd_type, num_points):
    for i in range(start, end):
        process_one_step(i, steps_dir[i], task_name, save_data_path, pcd_type, num_points)

def process_one_step(step_t, step, task_name, save_data_path, pcd_type, num_points):

    with open(f'{step}/other_data.pkl', 'rb') as f:
        data = pkl.load(f)

    images = {}
    depths = {}
    extrs = {}
    intrs = {}

    # TODO: 3 views for normal setting, 1 view for simple setting
    camera_name=['head', 'left', 'right']
    # cameras = ['head'] 

    H = int(480 / 2)
    W = int(640 / 2)

    for camera in cameras:
        color_file = f'{step}/{camera}_rgb.jpg'
        color = cv2.imread(color_file)[...,::-1]
        if H != 480:
            images[camera] = preproces_image(color, H, W, 'bicubic')
        else:
            images[camera] = color

        depth_file = f'{step}/{camera}_depth_x10000_uint16.png'
        depth_x10000_uint16 = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        depth = depth_x10000_uint16 / 10000
        if H != 480:
            depths[camera] = preproces_image(np.expand_dims(depth, axis=-1), H, W, 'nearest').squeeze(-1)
        else:
            depths[camera] = depth

        extrs[camera] = data['extr'][camera]
        intrs[camera] = data['intr'][camera]

    obs_pointcloud = get_pointcloud_from_multicameras(cameras, images, depths, extrs, intrs, H, W)
    obs_pointcloud = preprocess_point_cloud(obs_pointcloud, num_points, use_cuda=True)

    save_step_pcd_folder = f"{save_data_path}/episode{i}/{pcd_type}"
    os.makedirs(save_step_pcd_folder, exist_ok=True)
    save_step_pcd_path = f"{save_step_pcd_folder}/step{step_t:03d}.npy"
    np.save(save_step_pcd_path, obs_pointcloud)

if __name__ == "__main__":

    # TODO: change the task name
    task_name = "handover_and_insert_the_plate"
    # task_name = "wipe_the_plate" 
    # task_name = "press_the_bottle"
    # task_name = "scan_the_bottle"
    # task_name = 'wear_the_scarf' 

    expert_data_path = f'real_data/training_raw/{task_name}'
    # TODO: change the path, 'training_processed' for normal setting, 'training_processed_simple' for simple setting
    save_data_path = f'real_data/training_processed/point_cloud/{task_name}'
    
    # TODO: rgb_pcd_rps3072 for normal setting, rgb_pcd_rps512 for simple setting
    pcd_type = 'rgb_pcd_rps3072'

    num_points = int(pcd_type.split('ps')[-1])
    episodes_dir = sorted(glob.glob(f'{expert_data_path}/episode*'))

    os.makedirs(save_data_path, exist_ok=True)

    for i, episode_dir in enumerate(episodes_dir):
        steps_dir = sorted(glob.glob(f'{episode_dir}/steps/*'))
        demo_length = len(steps_dir)

        cprint('Processing {}'.format(episode_dir), 'green')

        threads = []
        threads_num = 20
        for thread_id in range(threads_num):
            start = thread_id * demo_length // threads_num
            end = (thread_id + 1) * demo_length // threads_num
            if thread_id == threads_num - 1:
                end = demo_length
            thread = threading.Thread(target=process_multi_steps, args=(steps_dir, start, end, task_name, save_data_path, pcd_type, num_points))
            threads.append(thread)
            thread.start()
    
        for thread in threads:
            thread.join()

            
        


