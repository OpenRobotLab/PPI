import roboticstoolbox as rtb
import numpy as np
from pathlib import Path
from pdb import set_trace
import pickle as pkl
import glob
from PIL import Image
import time
from ppi.policy.ppi import PPI
from ppi.model.vision.semantic_feature_extractor_bf16 import Fusion
import hydra
import torch
import torchvision
import pytorch3d.ops as torch3d_ops
from utils.transform_utils import xyzquat2xyzypr
import os
from segment_anything import SamPredictor, sam_model_registry
from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
import groundingdino.datasets.transforms as T
import matplotlib.pyplot as plt
import cv2

PI = np.pi
# TODO: modify your path to PPI
path_to_PPI = "/PATH/TO/PPI"

# HOME_POSE_L = [0.2658, -0.1535,  0.4866, 0., 0. ,0.]
# HOME_POSE_R = [0.2658,  0.1535,  0.4866, 0., 0. ,0.]

HOME_POSE_L = [ [1, 0, 0, 0.2658],
                [0, 1, 0, -0.1535],
                [0, 0, 1, 0.4866],
                [0, 0, 0, 1]]
HOME_POSE_R = [ [1, 0, 0, 0.2658],
                [0, 1, 0, 0.1535],
                [0, 0, 1, 0.4866],
                [0, 0, 0, 1]]

COLORS = [
    (144, 238, 144),  #  (LightGreen)
    (0, 255, 0),      #  (Green)
    (34, 139, 34),    #  (ForestGreen)
    (0, 100, 0)       #  (DarkGreen)
]

def preproces_image(image, img_size_H, img_size_W, mode):

    image = image.astype(np.float32)
    image = torch.from_numpy(image).cuda()
    image = image.permute(2, 0, 1) # HxWx4 -> 4xHxW

    # depth resize:  nearest 
    # rgb mask resize:  bicubic
    if mode == 'nearest':
        interpolation = torchvision.transforms.InterpolationMode.NEAREST
    elif mode == 'bicubic':
        interpolation = torchvision.transforms.InterpolationMode.BICUBIC

    image = torchvision.transforms.functional.resize(image, (img_size_H, img_size_W), interpolation)
    image = image.permute(1, 2, 0) # 4xHxW -> HxWx4
    image = image.cpu().numpy()
    return image

def random_point_sampling(points, num_points=1024, use_cuda=True):
    if use_cuda:
        points = torch.from_numpy(points).cuda()
    else:
        points = torch.from_numpy(points)

    num_total_points = points.shape[0]

    indices = torch.randperm(num_total_points)[:num_points]

    sampled_points = points[indices]

    if use_cuda:
        sampled_points = sampled_points.cpu().numpy()
    else:
        sampled_points = sampled_points.numpy()

    return sampled_points, indices.cpu().numpy()

def random_point_sampling_numpy(points, num_points=1024):
    num_total_points = points.shape[0]
    indices = np.random.permutation(num_total_points)[:num_points]
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

def preprocess_point_cloud(points, num_points, use_cuda=False):
    torch.cuda.synchronize()
    t_1 = time.time()

    WORK_SPACE = [
        [-0.1, 1.2],
        [-1, 1],
        [0.02, 1]
    ]
    
     # crop
    points = points[np.where((points[..., 0] > WORK_SPACE[0][0]) & (points[..., 0] < WORK_SPACE[0][1]) &
                                (points[..., 1] > WORK_SPACE[1][0]) & (points[..., 1] < WORK_SPACE[1][1]) &
                                (points[..., 2] > WORK_SPACE[2][0]) & (points[..., 2] < WORK_SPACE[2][1]))]

    points_xyz = points[..., :3]


    
    points_xyz, sample_indices = random_point_sampling_numpy(points_xyz, num_points)


    sample_indices = sample_indices
    points_rgb = points[sample_indices, 3:]
    points = np.hstack((points_xyz, points_rgb))

    torch.cuda.synchronize()
    t_2 = time.time()
    # print(f"Preprocess rps time: {t_2-t_1:.8f}s")
    return points

def get_pointcloud_from_multicameras(cameras, obs_hist, n_obs_steps, point_cloud_num, H, W):
    '''
        key: head_rgb_image, value: (2, 480, 640, 3)
        key: head_depth_image, value: (2, 480, 640)
        key: head_intr, value: (2, 3, 3)
        key: head_extr, value: (2, 4, 4)

        key: left_rgb_image, value: (2, 480, 640, 3)
        key: left_depth_image, value: (2, 480, 640)
        key: left_intr, value: (2, 3, 3)
        key: left_extr, value: (2, 4, 4)

        key: right_rgb_image, value: (2, 480, 640, 3)
        key: right_depth_image, value: (2, 480, 640)
        key: right_intr, value: (2, 3, 3)
        key: right_extr, value: (2, 4, 4)
    '''
    point_cloud_hist = np.zeros((n_obs_steps, point_cloud_num, 6))
    for step in range(n_obs_steps):
        multiview_pointcloud = None
        scale = H / 480
        for camera in cameras:
            torch.cuda.synchronize()
            t_before = time.time()
            depth_array = preproces_image(np.expand_dims(obs_hist[f'{camera}_depth_image'][step], axis=-1), H, W, 'nearest').squeeze(-1)
            rgb_array = preproces_image(obs_hist[f'{camera}_rgb_image'][step], H, W, 'bicubic')
            torch.cuda.synchronize()
            t_after = time.time()
            print(f"Preprocess {camera} image time: {t_after-t_before:.8f}s")

            intr = obs_hist[f'{camera}_intr'][step] * scale
            extr = obs_hist[f'{camera}_extr'][step]

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


        multiview_pointcloud_rps = preprocess_point_cloud(multiview_pointcloud, point_cloud_num)
        point_cloud_hist[step] = multiview_pointcloud_rps

    return point_cloud_hist

def process_gripper(gripper_open):
    if gripper_open > 0.5:
        return 1
    else:
        return 0

class InferenceController():
    def __init__(self, 
                 ckpt_path, 
                 control_type="pose", 
                 env=None, 
                 cfg=None, 
                 device=None, 
                 lang_emb_path=None, 
                 instruction=None, 
                 point_cloud_num=6144, 
                 pointflow_num=200, 
                 text_prompt=None, 
                 prompt_type='box', 
                 point_flow_sample_type='rps', 
                 sam_cameras=None, 
                 if_multi_objs=False, 
                 objs_and_pfl_num=None):
        
        self.ckpt_path = ckpt_path
        self.control_type = control_type
        self.env = env
        self.cfg = cfg
        self.device = device
        self.lang_emb_path = lang_emb_path
        self.instruction = instruction
        self.point_cloud_num = point_cloud_num
        self.fusion = Fusion(num_cam=3, feat_backbone='dinov2', device = self.device)
        self.pointflow_num = pointflow_num
        sam_checkpoint_path = f"{path_to_PPI}/real_world_deployment/repos/segment-anything/weights/sam_vit_b_01ec64.pth"
        sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint_path)
        sam.to(device='cuda')
        self.sam_predictor = SamPredictor(sam)
        gdino_config_path = f"{path_to_PPI}/real_world_deployment/repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"    #源码自带的配置文件
        gdino_checkpoint_path = f"{path_to_PPI}/real_world_deployment/repos/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth"   #下载的权重文件
        self.gdino_model = load_model(gdino_config_path, gdino_checkpoint_path)
        self.text_prompt = text_prompt
        self.prompt_type = prompt_type
        self.sample_type = point_flow_sample_type
        self.sam_cameras = sam_cameras
        self.if_multi_objs = if_multi_objs
        self.objs_and_pfl_num = objs_and_pfl_num

        self.is_first_frame = True
        self.result = None


        self.model: PPI = hydra.utils.instantiate(cfg.policy)
        ema_model_state_dict = torch.load(ckpt_path)['ema_model_state_dict']
        self.model.load_state_dict(ema_model_state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.query_num = 5
        self.query_id = 0
        self.result = None

    def reset(self):
        self.is_first_frame = True       
        self.query_id = 0
        self.result = None

    @torch.no_grad()
    def __call__(self, obs_hist, n_obs_steps, hist_num):
        torch.cuda.synchronize() 
        t1 = time.time()
        useful_obs = self.process_obs_hist(obs_hist, n_obs_steps, hist_num)

        torch.cuda.synchronize() 
        t2 = time.time()
        print(f"Process_obs_hist time: {t2-t1:.8f}s")

        torch.cuda.synchronize() 
        t3 = time.time()

        if self.query_id == 0:
            self.result = self.model.predict_action(useful_obs)

            torch.cuda.synchronize() 
            t4 = time.time()
            print(f"Predict_action time: {t4-t3:.8f}s")

        action = self.result['action'][0, self.query_id]
        self.query_id += 1

        if self.query_id == self.query_num:
            self.query_id = 0

        ####### visualize point flow #######
        point_flow = self.result['point_flow']
        image_w_point_flow = self.add_point_flow(obs_hist["head_rgb_image"][0], point_flow, obs_hist["head_extr"][0], obs_hist["head_intr"][0])
        cv2.imshow(f"Image with point flow", cv2.cvtColor(image_w_point_flow, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        ####### visualize point flow #######

        left_ee_pose = xyzquat2xyzypr(action[0:7].tolist())
        right_ee_pose = xyzquat2xyzypr(action[7:14].tolist())
        left_gripper_open = process_gripper(action[14])
        right_gripper_open = process_gripper(action[15])
        t5 = time.time()
        return left_ee_pose, right_ee_pose, left_gripper_open, right_gripper_open

    def add_point_flow(self, image, point_flow, extr, intr):
        """
        Project the 3D point cloud onto the image.

        Parasmeters:
            image (numpy.ndarray): [H, W, 3].
            point_flow (torch.Tensor): [1, N, 3].
            extr (numpy.ndarray): [4, 4].
            intr (numpy.ndarray): [3, 3].
        """
        device = point_flow.device
        extr = torch.tensor(extr, dtype=torch.float32, device=device)
        intr = torch.tensor(intr, dtype=torch.float32, device=device)

        N = point_flow.shape[1]
        ones = torch.ones(1, N, 1, device=device)
        point_flow_homogeneous = torch.cat([point_flow, ones], dim=-1)

        extr_inv = torch.inverse(extr)
        point_flow_camera = torch.matmul(extr_inv, point_flow_homogeneous.transpose(-1, -2)).transpose(-1, -2)

        point_flow_2d = torch.matmul(intr, point_flow_camera[:, :, :3].transpose(-1, -2)).transpose(-1, -2)

        point_flow_2d = point_flow_2d[:, :, :2] / point_flow_2d[:, :, 2:3]

        points_2d = point_flow_2d.cpu().numpy().reshape(-1, 2)

        for i, point in enumerate(points_2d):
            if point[0] == float('inf') or point[1] == float('inf') or point[0] == float('-inf') or point[1] == float('-inf'):
                continue
            x, y = int(point[0]), int(point[1])
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                if i < 200:
                    cv2.circle(image, (x, y), 2, COLORS[3], -1)
                elif i < 400:
                    cv2.circle(image, (x, y), 2, COLORS[2], -1)
                elif i < 600:
                    cv2.circle(image, (x, y), 2, COLORS[1], -1)
                elif i < 800:
                    cv2.circle(image, (x, y), 2, COLORS[0], -1)

        return image

    def process_obs_hist(self, obs_hist, n_obs_steps, hist_num=1):
        '''
        obs_hist = {"robot1_ee_pose": robot1_ee_pose, # obs['robot1_ee_pose']
                        "robot2_ee_pose": robot2_ee_pose, # obs['robot2_ee_pose']
                        "robot1_q_pose": robot1_q_pose, # obs['robot1_q_pose']
                        "robot2_q_pose": robot2_q_pose, # obs['robot2_q_pose']
                        "robot1_gripper_open": robot1_gripper_open, # obs['robot1_gripper_open']
                        "robot2_gripper_open": robot2_gripper_open, # obs['robot2_gripper_open']
                    }

        key: robot1_ee_pose, value: (2, 7)
        key: robot2_ee_pose, value: (2, 7)
        key: robot1_q_pose, value: (2, 7)
        key: robot2_q_pose, value: (2, 7)
        key: robot1_gripper_open, value: (2,)
        key: robot2_gripper_open, value: (2,)
        key: head_rgb_image, value: (2, 480, 640, 3)
        key: head_depth_image, value: (2, 480, 640)
        key: head_intr, value: (2, 3, 3)
        key: head_extr, value: (2, 4, 4)
        key: left_rgb_image, value: (2, 480, 640, 3)
        key: left_depth_image, value: (2, 480, 640)
        key: left_intr, value: (2, 3, 3)
        key: left_extr, value: (2, 4, 4)
        key: right_rgb_image, value: (2, 480, 640, 3)
        key: right_depth_image, value: (2, 480, 640)
        key: right_intr, value: (2, 3, 3)
        key: right_extr, value: (2, 4, 4)
        key: is_keyframe, value: (2,)
        key: instruction, value: (2,)
        
        obs_hist = {key: np.stack([value] * self.cfg.n_obs_steps) for key, value in current_obs.items()}
        '''
    

        torch.cuda.synchronize()
        t1 = time.time()

        agent_pos = None
        for i in range(1, hist_num + 1):
            robot1_ee_pose = torch.from_numpy(obs_hist['robot1_ee_pose'][-i]).to(self.device).reshape(1,7)
            robot2_ee_pose = torch.from_numpy(obs_hist['robot2_ee_pose'][-i]).to(self.device).reshape(1,7)
            robot1_gripper_open = torch.tensor(obs_hist['robot1_gripper_open'][-i], device=self.device).reshape(1,1)
            robot2_gripper_open = torch.tensor(obs_hist['robot2_gripper_open'][-i], device=self.device).reshape(1,1)

            if agent_pos is None:
                agent_pos = torch.cat(
                    [robot1_ee_pose, robot2_ee_pose,
                    robot1_gripper_open, robot2_gripper_open],
                    dim=-1
                )
            else:
                agent_pos = torch.cat(
                    [agent_pos, robot1_ee_pose, robot2_ee_pose,
                    robot1_gripper_open, robot2_gripper_open],
                    dim=-1
                )

        agent_pos = agent_pos.unsqueeze(0)

        torch.cuda.synchronize()
        t2 = time.time()
        print(f"Process agent_pos time: {t2-t1:.8f}s")

        cameras = ['head', 'left', 'right']
        # cameras = ['head'] # for simple-PPI
        H = int(480 / 2)
        W = int(640 / 2)
        point_cloud = get_pointcloud_from_multicameras(cameras, obs_hist, n_obs_steps, self.point_cloud_num, H, W)

        point_cloud = torch.from_numpy(point_cloud).to(self.device).unsqueeze(0)
        
        torch.cuda.synchronize()
        t3 = time.time()
        print(f"Process point_cloud time: {t3-t2:.8f}s")
        
        torch.cuda.synchronize()
        tcd1 = time.time()
        color = self.get_color(cameras, obs_hist, n_obs_steps)
        depth = self.get_depth(cameras, obs_hist, n_obs_steps)
        torch.cuda.synchronize()
        tcd2 = time.time()
        print(f"Get color & depth time: {tcd2-tcd1:.8f}s")

        torch.cuda.synchronize()
        tei1 = time.time()
        extrinsics = self.get_extr(cameras, obs_hist, n_obs_steps).to(self.device)
        intrinsics = self.get_intr(cameras, obs_hist, n_obs_steps).to(self.device)
        torch.cuda.synchronize()
        tei2 = time.time()
        print(f"Get extrinsics & intrinsics time: {tei2-tei1:.8f}s")

        torch.cuda.synchronize()
        t4 = time.time()
        print(f"Get camera information time: {t4-t3:.8f}s")

        dino_feature = self.get_dino_feature(point_cloud, color, depth, extrinsics, intrinsics)
        torch.cuda.synchronize()
        t5 = time.time()
        print(f"Process dino_feature time: {t5-t4:.8f}s")
# 
        if self.is_first_frame:
            self.is_first_frame = False
            if not self.if_multi_objs:
                self.initial_pointflow = self.get_initial_pointflow_single_obj(obs_hist)
            else:
                self.initial_pointflow = self.get_initial_pointflow_multi_objs(obs_hist)

        useful_obs = {
            'point_cloud': point_cloud,
            'agent_pos': agent_pos,
            'dino_feature': dino_feature,
            'initial_point_flow': self.initial_pointflow
        }

        if os.path.exists(self.lang_emb_path):
            with open(self.lang_emb_path, 'rb') as f:
                instruction_embedding_dict = pkl.load(f)

            useful_obs['lang'] = instruction_embedding_dict[obs_hist['instruction'][-1]].reshape(1, 1, 1024)

        torch.cuda.synchronize()
        t6 = time.time()
        print(f"Process obs hist left time: {t6-t5:.8f}s")

        return useful_obs
    
    def get_dino_feature(self, ptc, all_cam_images, depth_image_lst, cameras_extrinsics, cameras_intrinsics):
        torch.cuda.synchronize()
        get_dino_feature_1=time.time()

        ptc = ptc[..., :3]
        B = ptc.shape[0]
        T = ptc.shape[1]
        dino_feature = torch.zeros((B, T, self.point_cloud_num, 384))
        for b in range(B):
            for t in range(T):
                pointcloud = ptc[b,t]
                obs ={
                    'color': all_cam_images[b,t],
                    'depth': depth_image_lst[b,t],
                    'pose': cameras_extrinsics[b,t],
                    'K': cameras_intrinsics[b,t]
                }

                dino_feature[b, t] = self.fusion.extract_semantic_feature_from_ptc(pointcloud, obs)


        torch.cuda.synchronize()
        get_dino_feature_2=time.time()
        return dino_feature.to(device=self.device)
    
    def get_color(self, cameras, obs_hist, n_obs_steps):
        color = torch.zeros((1, n_obs_steps, len(cameras), 480, 640, 3), device=self.device)
        for i, camera in enumerate(cameras):
            torch.cuda.synchronize()
            tcd1 = time.time()

            camera_data = torch.as_tensor(obs_hist[f'{camera}_rgb_image'], device=self.device)

            torch.cuda.synchronize()
            tcd2 = time.time()
            color[0, :, i] = camera_data
            print(f"torch.from_numpy color time: {tcd2-tcd1:.8f}s")
        return color
    
    def get_depth(self, cameras, obs_hist, n_obs_steps):
        depth = torch.zeros((1, n_obs_steps, len(cameras), 480, 640)).to(self.device)
        for i, camera in enumerate(cameras):
            torch.cuda.synchronize()
            tcd1 = time.time()
            depth[0, :, i] = torch.as_tensor(obs_hist[f'{camera}_depth_image'], device=self.device)
            torch.cuda.synchronize()
            tcd2 = time.time()
            print(f"torch.from_numpy time: {tcd2-tcd1:.8f}s")
        return depth
    
    def get_extr(self, cameras, obs_hist, n_obs_steps):
        extr = torch.zeros((1, n_obs_steps, len(cameras), 3, 4)).to(self.device)
        for i, camera in enumerate(cameras):
            extr[0, :, i] = torch.from_numpy(obs_hist[f'{camera}_extr'][..., :3, :])
        return extr
    
    def get_intr(self, cameras, obs_hist, n_obs_steps):
        intr = torch.zeros((1, n_obs_steps, len(cameras), 3, 3)).to(self.device)
        for i, camera in enumerate(cameras):
            intr[0, :, i] = torch.from_numpy(obs_hist[f'{camera}_intr'])
        return intr
    
    def get_initial_pointflow_single_obj(self, obs_hist):
        point_clouds = []
        for camera in self.sam_cameras:
            image = obs_hist[f"{camera}_rgb_image"][0]

            point_coords=self.get_point_from_mask(image)
            depth = obs_hist[f'{camera}_depth_image']
            depth_image_m = depth[0]
            extrinsics = obs_hist[f'{camera}_extr']  # overhead_camera_extrinsics, shape: torch.Size([1, 2, 4, 4])   
            extrinsics = extrinsics[0]
            intrinsics = obs_hist[f'{camera}_intr']
            intrinsics = intrinsics[0]
            pc_initial = self.pointflow_from_tracks(depth_image_m, extrinsics, intrinsics, point_coords)
            point_clouds.append(pc_initial)
        
        point_clouds = np.concatenate(point_clouds,axis=0)
        total_points = point_clouds.shape[0]
        point_clouds = point_clouds.reshape(1,total_points,3)
        point_clouds = torch.tensor(point_clouds)
        if self.sample_type=='fps':
            sampled_pc, idx = torch3d_ops.sample_farthest_points(point_clouds, K=self.pointflow_num)
            sampled_point_clouds = sampled_pc[0][idx.sort()[1][0]]
            return sampled_point_clouds.unsqueeze(0).unsqueeze(0).to(self.device)
        elif self.sample_type == 'rps':
            rand_idx = np.random.choice(total_points, int(self.pointflow_num), replace=False)
            rand_idx = sorted(rand_idx)
            sampled_point_clouds = point_clouds[:, rand_idx,:]
            return sampled_point_clouds.unsqueeze(0).to(self.device)
        
    def get_initial_pointflow_multi_objs(self, obs_hist):
        point_clouds = {}
        
        for obj_name in self.objs_and_pfl_num.keys():
            point_clouds[obj_name] = []

        for camera in self.sam_cameras:
            image = obs_hist[f"{camera}_rgb_image"][0]

            point_coords=self.get_point_from_mask_multi_objs(image)    

            depth = obs_hist[f'{camera}_depth_image']
            depth_image_m = depth[0]
            extrinsics = obs_hist[f'{camera}_extr']
            extrinsics = extrinsics[0]
            intrinsics = obs_hist[f'{camera}_intr']
            intrinsics = intrinsics[0]
            
            pc_initial_multi_objs = {}
            for obj_name, point_coord in point_coords.items():
                this_pc_initial = self.pointflow_from_tracks(depth_image_m, extrinsics, intrinsics, point_coord)
                pc_initial_multi_objs[obj_name] = this_pc_initial
                point_clouds[obj_name].append(this_pc_initial)
        
        init_sampled_pointflow = []

        for obj_name, point_cloud in point_clouds.items():
            point_clouds[obj_name] = np.concatenate(point_cloud,axis=0)

            total_points = point_clouds[obj_name].shape[0]
            point_clouds[obj_name] = point_clouds[obj_name].reshape(1,total_points,3)
            point_clouds[obj_name] = torch.tensor(point_clouds[obj_name])
            if self.sample_type=='fps':
                sampled_pc, idx = torch3d_ops.sample_farthest_points(point_clouds[obj_name], K=self.objs_and_pfl_num[obj_name])
                sampled_point_clouds = sampled_pc[0][idx.sort()[1][0]].unsqueeze(0).unsqueeze(0).to(self.device)
                init_sampled_pointflow.append(sampled_point_clouds)
            elif self.sample_type == 'rps':
                rand_idx = np.random.choice(total_points, self.objs_and_pfl_num[obj_name], replace=False)
                rand_idx = sorted(rand_idx)
                sampled_point_clouds = point_clouds[obj_name][:, rand_idx,:].unsqueeze(0).to(self.device)
                init_sampled_pointflow.append(sampled_point_clouds)

        combined_init_sampled_pointflow = torch.cat(init_sampled_pointflow, dim=2)

        return combined_init_sampled_pointflow
        
    def get_point_from_mask(self, image):
        device = 'cuda'

        ############################### GroundingDINO ###############################
        image_transformed=self.transform_image(image)
        boxes, logits, phrases = predict(model=self.gdino_model,
                                         image=image_transformed,
                                         caption=self.text_prompt,
                                         box_threshold=0.35,
                                         text_threshold=0.8,
                                         device=device)

        boxes = boxes[0].reshape(1,4)
        H, W, _ = image.shape
        boxes_np = boxes.cpu().numpy() * np.array([W, H, W, H])
        cx, cy, wx, wy = boxes_np[:, 0], boxes_np[:, 1], boxes_np[:, 2], boxes_np[:, 3]
        prompt_boxes =  np.array([cx - wx/2, cy - wy/2, cx + wx/2, cy + wy/2]).T
        prompt_boxes = prompt_boxes.astype(int)
        input_boxes = torch.tensor(prompt_boxes[0]).to(self.sam_predictor.device)
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        prompt_point = np.array([cx[0], cy[0]])

        x_min, y_min, x_max, y_max = cx - wx/2, cy - wy/2, cx + wx/2, cy + wy/2

        ############################### GroundingDINO ###############################
        
        ############################### SAM ###############################
        input_label = np.array([1])
        self.sam_predictor.set_image(image)
        if self.prompt_type=="point":
            mask, score, logit = self.sam_predictor.predict(point_coords=prompt_point.reshape(1,2),
                                                    point_labels=input_label,
                                                    multimask_output=False)
        elif self.prompt_type=="box":
            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            mask = masks[0].cpu().numpy()
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1)
        true_coords = np.argwhere(mask_image[:, :, 0])
        plt.figure(figsize=(10,10))
        plt.imshow(image/255)
        self.show_mask(mask, plt.gca())
        self.show_points(true_coords, plt.gca(),color='green')
        plt.axis('on')
        folder_path = f'{path_to_PPI}/real_world_deployment/tests/check_point_flow_6d/test_time_check_sampled_points'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.savefig(os.path.join(folder_path,f"ample_points.png"))
        plt.close
        return true_coords

    def get_point_from_mask_multi_objs(self, image):
        device = 'cuda'

        ############################### GroundingDINO ###############################
        image_transformed=self.transform_image(image)


        transformed_boxes = []
        for text_prompt in self.text_prompt:
            box, logits, phrases = predict(model=self.gdino_model,
                                         image=image_transformed,
                                         caption=text_prompt,
                                         box_threshold=0.35,
                                         text_threshold=0.8,
                                         device=device)

            box = box[0].reshape(1,4)
            H, W, _ = image.shape
            box_np = box.cpu().numpy() * np.array([W, H, W, H])
            cx, cy, wx, wy = box_np[:, 0], box_np[:, 1], box_np[:, 2], box_np[:, 3]
            prompt_box =  np.array([cx - wx/2, cy - wy/2, cx + wx/2, cy + wy/2]).T
            prompt_box = prompt_box.astype(int)
            input_box = torch.tensor(prompt_box[0]).to(self.device)
            transformed_box = self.sam_predictor.transform.apply_boxes_torch(input_box, image.shape[:2])

            transformed_boxes.append(transformed_box)
        ############################### GroundingDINO ###############################
        
        ############################### SAM ###############################

        true_coords_multi = {}

        objs_name_list = list(self.objs_and_pfl_num.keys())

        for i, sam_prompt in enumerate(transformed_boxes):
            self.sam_predictor.set_image(image)

            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=sam_prompt,
                multimask_output=False,
            )
            mask = masks[0].cpu().numpy()

            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1)
            true_coords = np.argwhere(mask_image[:, :, 0]) 
            true_coords_multi[objs_name_list[i]] = true_coords

        true_coords_vis = np.concatenate([true_coords_multi[obj_name] for obj_name in objs_name_list], axis=0)
        plt.figure(figsize=(10,10))
        plt.imshow(image/255)
        self.show_mask(mask, plt.gca())
        self.show_points(true_coords_vis, plt.gca(),color='green')
        plt.axis('on')
        folder_path = f'{path_to_PPI}/real_world_deployment/tests/check_point_flow_6d/test_time_check_sampled_points'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.savefig(os.path.join(folder_path,f"ample_points.png"))
        plt.close

        return true_coords_multi


    def transform_image(self, image):
        transform = T.Compose([T.RandomResize([800], max_size=1333),T.ToTensor(),T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        pil_image = Image.fromarray(np.uint8(image))
        image_transformed, _ = transform(pil_image, None)
        return image_transformed
    
    def show_mask(self,mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
        
    def show_points(self,coords, ax, color):
        ax.scatter(coords[:, 1], coords[:, 0], color=color, marker='*', s=1, edgecolor='white', linewidth=0.5)

    def pointflow_from_tracks(
            self, 
            depth, extrinsics,
            intrinsics, pixel_tracks):
        """
        pixel_tracks: Npts, 2
        depth:256,256
        Converts depth (in meters) to point cloud in word frame.
        Return: A numpy array of size (width, height, 3)
        """
        # set_trace()
        Npts,_ = pixel_tracks.shape
        pixel_tracks = pixel_tracks.astype(int)
        ones_column = np.ones((Npts,1))
        upc = np.hstack((pixel_tracks[:,[1,0]],ones_column))
        depth_values = depth[pixel_tracks[:,0],pixel_tracks[:,1]].reshape(-1,1)
        pc = upc * depth_values
        #pc: Npts,3
        C = np.expand_dims(extrinsics[:3, 3], 0).T
        R = extrinsics[:3, :3] 
        R_inv = R.T  # inverse of rot matrix is transpose
        R_inv_C = np.matmul(R_inv, C)
        extrinsics = np.concatenate((R_inv, -R_inv_C), -1) # 3*4
        cam_proj_mat = np.matmul(intrinsics, extrinsics) # 3*4
        cam_proj_mat_homo = np.concatenate(
            [cam_proj_mat, [np.array([0, 0, 0, 1])]]) # 4*4
        cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)[0:3]
        world_coords_homo = self._2dpixel_to_world_coords(pc, cam_proj_mat_inv)
        world_coords = world_coords_homo[..., :-1] # (Npts, 3)

        return world_coords
        
    def _2dpixel_to_world_coords(self, pixel_coords, cam_proj_mat_inv):
        
        Npts = pixel_coords.shape[0]
        pixel_coords = np.concatenate(
            [pixel_coords, np.ones((Npts, 1))], -1)
        world_coords = self._2dtransform(pixel_coords, cam_proj_mat_inv)
        world_coords_homo = np.concatenate(
            [world_coords, np.ones((Npts, 1))], axis=-1)
        return world_coords_homo
    
    def _2dtransform(self, coords, trans):
        Npts = coords.shape[0]
        coords = np.transpose(coords, (1, 0))
        transformed_coords_vector = np.matmul(trans, coords)
        transformed_coords_vector = np.transpose(
            transformed_coords_vector, (1, 0))
        return np.reshape(transformed_coords_vector,
                        (Npts, -1))
    