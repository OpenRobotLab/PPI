import copy
import logging
from functools import lru_cache
import pickle
import os
from typing import List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from yarr.agents.agent import Agent, Summary, ActResult, \
    ScalarSummary, HistogramSummary
from helpers import utils
from helpers.utils import stack_on_channel
import zarr
import time

from pdb import set_trace
from PIL import Image
from diffusion_policy_3d.model.vision.semantic_feature_extractor import Fusion
import matplotlib.pyplot as plt
from pdb import set_trace
from pyrep.objects.shape import Shape 
from pyrep.const import PrimitiveShape
from pyrep.objects.vision_sensor import VisionSensor

import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
from segment_anything import SamPredictor, sam_model_registry
import matplotlib.patches as patches
from pytorch3d.ops import sample_farthest_points

class PPIAgent(Agent):

    def __init__(self,
                 actor_network: nn.Module,
                 cameras: List[str],
                 task_name=None,
                 weight_name=None,
                 fps_num=512,
                 cameras_pcd=None,
                 use_pc_color=False,
                 use_lang=False,
                 bounding_box=None,
                 episode_length = None,
                 prediction_type = None,
                 save_path = None,
                 query_freq = 1,
                 horizon_continuous = 3,
                 horizon_keyframe = 0,
                 predict_point_flow=True,
                 pointflow_num=200,
                 text_prompt=None,
                 prompt_type=None,
                 sample_type=None,
                 sam_cameras=None,
                 ckpt_name=None,
                 jump_step=1,
                 sam_checkpoint_path="pretrained_models/sam_vit_b_01ec64.pth",
                 gdino_config_path="repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
                 gdino_checkpoint_path="pretrained_models/groundingdino_swinb_cogcoor.pth",
                 instruction_embeddings_path="data/training_processed/instruction_embeddings.pkl"
                 ):
        self.cameras = cameras
        self._actor = actor_network
        self.task_name = task_name
        self.weight_name = weight_name
        self.fps_num = fps_num
        self.cameras_pcd = cameras_pcd
        self.use_pc_color = use_pc_color
        self.use_lang = use_lang
        self.bounding_box = bounding_box
        self._episode_length = episode_length
        self.prediction_type = prediction_type
        self._episode=None
        self.save_path = save_path
        self.query_freq = query_freq
        self.action_id = 0
        self.result_action=None
        self.result_object=None
        self.horizon_continuous = horizon_continuous
        self.horizon_keyframe = horizon_keyframe
        self.visual_targets = []
        self.pointflow_num = pointflow_num
        self.predict_point_flow=predict_point_flow
        sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint_path)
        sam.to(device='cuda')
        self.sam_predictor = SamPredictor(sam)
        self.gdino_model = load_model(gdino_config_path, gdino_checkpoint_path)
        self.text_prompt = text_prompt
        self.prompt_type = prompt_type
        self.sample_type = sample_type
        self.sam_cameras = sam_cameras
        with open(instruction_embeddings_path,"rb") as f:
            self.text_embedding_list = pickle.load(f)
        self.ckpt_name = ckpt_name
        self.jump_step=jump_step
        
        
    def build(self, training: bool, device: torch.device = None):
        if device is None:
            device = torch.device("cpu")
        self._actor = self._actor.to(device).train(training)
        self._device = device
        self.fusion = Fusion(num_cam=6,feat_backbone='dinov2',device = self._device)

    def reset(self):
        if self._episode == None:
            self._episode = 0
        else:
            self._episode += 1
        self._timestep = 0
        self.action_id = 0
        self.result_action=None
        self.result_object=None
        
    def get_initial_pointflow(self, fps, observation):
        point_clouds = []
        for camera in self.sam_cameras:
            image = observation['%s_rgb' % camera].permute(0, 1, 3, 4, 2).cpu().numpy()
            image = image[0,0]
            point_coords=self.get_point_from_mask(image)
            if not point_coords.all()==0:
                depth = observation[f'{camera}_depth'].cpu().numpy()
                depth_image_m = depth[0,0,0]
                extrinsics = observation[f'{camera}_camera_extrinsics']
                extrinsics = extrinsics[:,:,:3]
                extrinsics = extrinsics.cpu().numpy()[0,0]
                intrinsics = observation[f'{camera}_camera_intrinsics'].cpu().numpy()
                intrinsics = intrinsics[0,0]
                pc_initial = self.pointflow_from_tracks(depth_image_m, extrinsics,intrinsics, point_coords)
                point_clouds.append(pc_initial)
            else: 
                camera = "wrist_right"
                image = observation['%s_rgb' % camera].permute(0, 1, 3, 4, 2).cpu().numpy()
                image = image[0,0]
                point_coords=self.get_point_from_mask(image)
                depth = observation[f'{camera}_depth'].cpu().numpy()
                depth_image_m = depth[0,0,0]
                extrinsics = observation[f'{camera}_camera_extrinsics'] 
                extrinsics = extrinsics[:,:,:3]
                extrinsics = extrinsics.cpu().numpy()[0,0]
                intrinsics = observation[f'{camera}_camera_intrinsics'].cpu().numpy()
                intrinsics = intrinsics[0,0]
                pc_initial = self.pointflow_from_tracks(depth_image_m, extrinsics,intrinsics, point_coords)
                point_clouds.append(pc_initial)
        
        point_clouds = np.concatenate(point_clouds,axis=0)
        total_points = point_clouds.shape[0]
        point_clouds = point_clouds.reshape(1,total_points,3)
        point_clouds = torch.tensor(point_clouds)
        if self.sample_type=='fps':
            sampled_pc, idx = sample_farthest_points(point_clouds, K=fps)
            sampled_point_clouds = sampled_pc[0][idx.sort()[1][0]]
            return sampled_point_clouds.unsqueeze(0).unsqueeze(0).to(self._device)
        elif self.sample_type == 'rps':
            rand_idx = np.random.choice(total_points, int(fps), replace=False)
            rand_idx = sorted(rand_idx)
            sampled_point_clouds = point_clouds[:, rand_idx,:]
            return sampled_point_clouds.unsqueeze(0).to(self._device)
    
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
        Npts,_ = pixel_tracks.shape
        pixel_tracks = pixel_tracks.astype(int)
        ones_column = np.ones((Npts,1))
        upc = np.hstack((pixel_tracks[:,[1,0]],ones_column))
        depth_values = depth[pixel_tracks[:,0],pixel_tracks[:,1]].reshape(-1,1)
        pc = upc * depth_values
        C = np.expand_dims(extrinsics[:3, 3], 0).T
        R = extrinsics[:3, :3]
        R_inv = R.T
        R_inv_C = np.matmul(R_inv, C)
        extrinsics = np.concatenate((R_inv, -R_inv_C), -1)
        cam_proj_mat = np.matmul(intrinsics, extrinsics)
        cam_proj_mat_homo = np.concatenate(
            [cam_proj_mat, [np.array([0, 0, 0, 1])]])
        cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)[0:3]
        world_coords_homo = self._2dpixel_to_world_coords(pc, cam_proj_mat_inv)
        world_coords = world_coords_homo[..., :-1]
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
    
    def transform_image(self,image):
        transform = T.Compose([T.RandomResize([800], max_size=1333),T.ToTensor(),T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        pil_image = Image.fromarray(np.uint8(image))
        image_transformed, _ = transform(pil_image, None)
        return image_transformed
    
    def get_point_from_mask(self, image):
        device = 'cuda'
        image_transformed=self.transform_image(image)
        boxes, logits, phrases = predict(model=self.gdino_model,
                                         image=image_transformed,
                                         caption=self.text_prompt,
                                         box_threshold=0.35,
                                         text_threshold=0.45,
                                         device=device)
        if len(boxes)==0:
            return np.array([0,0])
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
        return true_coords
    
    
    def preprocess_images(self, replay_sample: dict):
        stacked_rgb = []
        # set_trace()

        for camera in self.cameras:
            rgb = replay_sample['%s_rgb' % camera]
            stacked_rgb.append(rgb)

        stacked_rgb = torch.stack(stacked_rgb, dim=2)
        return stacked_rgb
    
    def preprocess_pcd(self, observation: dict):
        pc = None
        for camera in self.cameras_pcd:
            B, T, C, H, W = observation[f'{camera}_point_cloud'].shape
            precent_pc = observation[f'{camera}_point_cloud'].view(B, T, C, H*W).permute(0, 1, 3, 2)
            rgb = torch.round((observation[f'{camera}_rgb'] + 1) * 255 / 2)
            precent_rgbpc = torch.cat([observation[f'{camera}_point_cloud'], rgb], dim=2).view(B, T, C*2, H*W).permute(0, 1, 3, 2)
            if pc is None:
                if self.use_pc_color:
                    pc = precent_rgbpc
                else:
                    pc = precent_pc
            else:
                if self.use_pc_color:
                    pc = torch.cat([pc, precent_rgbpc], dim=-2)
                else:
                    pc = torch.cat([pc, precent_pc], dim=-2)
        if self.use_pc_color:
            point_cloud = torch.zeros((B, T, self.fps_num, C*2))
        else:
            point_cloud = torch.zeros((B, T, self.fps_num, C))

        device = pc.device
        bounding_box = torch.tensor(self.bounding_box).to(device)
        for b in range(B):
            for t in range(T):
                point_cloud[b, t] = self.RPS_with_bounding_box(pc[b, t], self.fps_num, bounding_box)
        return point_cloud.to(self._device)
    
    def preprocess_depth(self, observation: dict):
        stacked_depth = []
        for camera in self.cameras:
            depth = observation[f'{camera}_depth'].squeeze(2)
            stacked_depth.append(depth)
        stacked_depth = torch.stack(stacked_depth, dim=2)
        return stacked_depth
    
    def transform_camera_extrinsics(self,extrinsics):
        C = extrinsics[...,:3, 3:]
        R = extrinsics[...,:3, :3]
        R_inv = R.transpose(-1,-2)
        R_inv_C = torch.matmul(R_inv, C)
        output_extrinsics = torch.cat((R_inv, -R_inv_C), dim=-1)
        return output_extrinsics
    
    def preprocess_extrinsics_intrinsics(self, observation: dict):
        stacked_extrinsics = []
        stacked_intrinsics = []
        for camera in self.cameras:
            extrinsics = observation[f'{camera}_camera_extrinsics']
            extrinsics = extrinsics[:,:,:3]
            extrinsics = self.transform_camera_extrinsics(extrinsics)
            stacked_extrinsics.append(extrinsics)
            intrinsics = observation[f'{camera}_camera_intrinsics']
            stacked_intrinsics.append(intrinsics)
        stacked_extrinsics = torch.stack(stacked_extrinsics, dim=2)
        stacked_intrinsics = torch.stack(stacked_intrinsics, dim=2)
        return stacked_extrinsics, stacked_intrinsics
    
    def get_dino_feature(self, ptc, all_cam_images, depth_image_lst, cameras_extrinsics, cameras_intrinsics):
        B = ptc.shape[0]
        T = ptc.shape[1]
        dino_feature = torch.zeros((B, T, self.fps_num, 384))
        all_cam_images = all_cam_images.permute(0, 1, 2, 4, 5, 3)
        for b in range(B):
            for t in range(T):
                pointcloud = ptc[b,t]
                obs ={
                    'color': all_cam_images[b,t],
                    'depth': depth_image_lst[b,t],
                    'pose': cameras_extrinsics[b,t],
                    'K': cameras_intrinsics[b,t]
                }
                torch.cuda.synchronize()
                fusion_function_end=time.time()
                dino_feature[b, t] = self.fusion.extract_semantic_feature_from_ptc(pointcloud,obs)
                torch.cuda.synchronize()
                time2 = time.time()
        return dino_feature.to(device=self._device)
    
    def visualizer(self, actions):
        keyframe_steps = self.horizon_keyframe
        continuous_steps = self.horizon_continuous-1
        steps = keyframe_steps + continuous_steps
        if not self.visual_targets:
            for arm in range(2):
                targets = []
                for target in range(steps):
                    if self.prediction_type=="keyframe_continuous":
                        if target<continuous_steps:
                            vis_color = [1 * (target + 1)/steps,0,0]
                        else:
                            vis_color = [0,0,1 * (target + 1)/steps]
                    elif self.prediction_type=="continuous":
                        vis_color = [1 * (target + 1)/steps,0,0]
                    elif self.prediction_type=="keyframe":
                        vis_color = [0,0,1 * (target + 1)/steps]
                    sphere = Shape.create(
                        type=PrimitiveShape.SPHERE,
                        color=vis_color,  
                        size=[0.01, 0.01, 0.01],
                        position=actions[target][7 * arm : 3 + 7 * arm]
                    )
                    sphere.set_dynamic(False)
                    sphere.set_respondable(False)
                    sphere.set_renderable(True)
                    targets.append(sphere)
                self.visual_targets.append(targets)
        else:
            for arm in range(2):
                for target in range(steps):
                    self.visual_targets[arm][target].set_position(actions[target][7 * arm : 3 + 7 * arm])
                    self.visual_targets[arm][target].set_renderable(True)
                    
    
    def quaternion_to_euler(self,q):
        w, x, y, z = q
        pitch = np.arcsin(2 * (w * y - z * x))
        pitch = np.clip(pitch, -np.pi / 2, np.pi / 2)
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return np.array([roll, pitch, yaw])
                    
    def visualizer_object_pose_pointflow(self, actions, object_pose=None, point_flow=None):
        keyframe_steps = self.horizon_keyframe
        continuous_steps = self.horizon_continuous-1
        steps = keyframe_steps + continuous_steps
        if point_flow!=None:
            point_num = int(len(point_flow)/keyframe_steps)
        if not self.visual_targets:
            for arm in range(2):
                targets = []
                for target in range(steps):
                    if target<continuous_steps:
                        vis_color = [1 * (target + 1)/steps,0,0]
                    else:
                        vis_color = [0,0,1 * (target + 1)/steps]
                    sphere = Shape.create(
                        type=PrimitiveShape.SPHERE,
                        color=vis_color,  
                        size=[0.01, 0.01, 0.01],
                        position=actions[target][7 * arm : 3 + 7 * arm]
                    )
                    sphere.set_dynamic(False)
                    sphere.set_respondable(False)
                    sphere.set_renderable(True)
                    targets.append(sphere)
                self.visual_targets.append(targets)
            targets = []
            if self.predict_point_flow:
                targets = []
                for step in range(keyframe_steps):
                    for point in range(point_num):
                        vis_color = [0,1 * (step + 1)/keyframe_steps,0]
                        sphere = Shape.create(
                            type=PrimitiveShape.SPHERE,
                            color=vis_color,  
                            size=[0.01, 0.01, 0.01],
                            position=point_flow[step*point_num+point][: 3]
                        )
                        sphere.set_dynamic(False)
                        sphere.set_respondable(False)
                        sphere.set_renderable(True)
                        targets.append(sphere)
                self.visual_targets.append(targets)
                        
        else:
            for arm in range(2):
                for target in range(steps):
                    self.visual_targets[arm][target].set_position(actions[target][7 * arm : 3 + 7 * arm])
                    self.visual_targets[arm][target].set_renderable(True)
            id=1
            if self.predict_point_flow:
                id=id+1
                for target in range(keyframe_steps):
                    for point in range(point_num):
                        self.visual_targets[id][target*point_num+point].set_position(point_flow[target*point_num+point][: 3])
                        self.visual_targets[id][target*point_num+point].set_renderable(True)
        
    def act(self, step: int, observation: dict,
            deterministic=False) -> ActResult:
        torch.cuda.synchronize()
        start_time = time.time()
        useful_obs = dict()
        query_freq = self.query_freq
        left_gripper_pose = observation['left_gripper_pose']
        right_gripper_pose = observation['right_gripper_pose']
        left_gripper_open = observation['left_gripper_open']
        right_gripper_open = observation['right_gripper_open']
        agent_pos = torch.cat(
            [left_gripper_pose, right_gripper_pose, 
            left_gripper_open, right_gripper_open],
            dim=-1
        )
        point_cloud = self.preprocess_pcd(observation)
        if self.use_lang:
            # set_trace()
            if self._timestep==0:
                self.lang_goal = observation.get('lang_goal', None)
                # if self.task_name=="bimanual_handover_item":
                #     colors_map = {
                #         "yellow": "a small lemon yellow cube",
                #         "green": "a small bright green cube",
                #         "blue": "a small blue cube",
                #         "purple": "a small dark purple cube",
                #         "red": "a small bright red cube"
                #     }
                #     color_name = self.lang_goal.split()[3]
                #     self.text_prompt = f'{colors_map[color_name]}'
        
        if self._timestep==0 or self._timestep==1:
            self.initial_pointflow = self.get_initial_pointflow(self.pointflow_num, observation)            
        torch.cuda.synchronize()
        ptc_time = time.time()
        depth = self.preprocess_depth(observation)
        extrinsics, intrinsics = self.preprocess_extrinsics_intrinsics(observation)
        color = self.preprocess_images(observation)
        torch.cuda.synchronize()
        dino_start_time=time.time()
        dino_feature = self.get_dino_feature(point_cloud,color,depth,extrinsics,intrinsics)
        torch.cuda.synchronize()
        dino_time = time.time()

        useful_obs = {
            'point_cloud': point_cloud,
            'agent_pos': agent_pos,
            'dino_feature': dino_feature,
            'initial_point_flow': self.initial_pointflow
        }

        useful_obs['lang'] = torch.from_numpy(self.text_embedding_list[self.lang_goal]).unsqueeze(0).unsqueeze(0).to(device=self._device)
        if self._timestep % query_freq == 0 or self._timestep==1:
            with torch.no_grad():
                result_dict = self._actor.predict_action(useful_obs)
                self.result_action = result_dict['action'].to(device=self._device)
                if self.predict_point_flow:
                    self.result_pointflow = result_dict['point_flow_pred'].to(device=self._device)
        if self.prediction_type == "continuous":
            result = self.result_action[0, self.action_id+1].to(device=self._device)
        elif self.prediction_type == "keyframe":
            result = self.result_action[0, self.action_id].to(device=self._device)
        elif self.prediction_type == "keyframe_continuous":
            result = self.result_action[0, self.action_id+1].to(device=self._device)

        if self.action_id < self.jump_step*(query_freq-1):
            self.action_id += self.jump_step
        else:
            self.action_id = 0

        quat1 = result[10:14]
        quat1_normalized = quat1 / quat1.norm(dim=-1, keepdim=True)

        quat2 = result[3:7]
        quat2_normalized = quat2 / quat2.norm(dim=-1, keepdim=True)

        raw_action = torch.cat([
            result[7:10],                    
            quat1_normalized,                
            torch.clamp(result[15].unsqueeze(-1),min=0.0,max=1.0),
            torch.tensor([1,]).to(device=self._device),
            result[0:3],                     
            quat2_normalized,                
            torch.clamp(result[14].unsqueeze(-1),min=0.0,max=1.0),
            torch.tensor([1,]).to(device=self._device)
        ], dim=-1)
        torch.cuda.synchronize()
        one_prediction_time = time.time()
            
        if self.predict_point_flow:
            self.visualizer_object_pose_pointflow(actions=self.result_action[0,1:].tolist(), point_flow = self.result_pointflow[0].tolist()) 
        else:
            if self.prediction_type == "keyframe":
                self.visualizer(self.result_action[0].tolist())
            else:
                self.visualizer(self.result_action[0,1:].tolist())
        self._timestep+=1
        print(f"timestep:{self._timestep}")
        return ActResult(raw_action.detach().cpu().numpy(),visual_targets = self.visual_targets)
            
    

    def act_summaries(self) -> List[Summary]:
        return []

    def load_weights(self, savedir: str):
        ckpt_dir=os.path.join(savedir, self.weight_name, 'checkpoints', f'{self.ckpt_name}.pth.tar')              
        checkpoint_dict = torch.load(ckpt_dir)       
        self._actor.load_state_dict(checkpoint_dict['model_state_dict'])
        print('Loaded weights from %s' % ckpt_dir)

    def update(self, step: int, replay_sample: dict) -> dict:
        pass

    def update_summaries(self) -> List[Summary]:
        pass

    def FPS_with_bounding_box(self, pc, num_samples, bounding_box):
        """
        Perform farthest point sampling (FPS) on the point cloud within a bounding box using PyTorch.
        
        Args:
            pc: PyTorch tensor of shape (N, 3), where N is the number of points.
            num_samples: Number of points to sample.
            bounding_box: PyTorch tensor of shape (2, 3), where the first row contains the
                        minimum x, y, z values and the second row contains the
                        maximum x, y, z values of the bounding box.
        
        Returns:
            sampled_pc: PyTorch tensor of shape (num_samples, 3), sampled points within the bounding box.
        """
        if num_samples >= pc.shape[0]:
            return pc

        # Extract the min and max corners of the bounding box
        min_corner = bounding_box[0]
        max_corner = bounding_box[1]

        # Filter points to only include those within the bounding_box
        pc_xyz = pc[:, :3]
        inside_mask = (pc_xyz >= min_corner) & (pc_xyz <= max_corner)
        inside_mask = inside_mask[:, 0] & inside_mask[:, 1] & inside_mask[:, 2]
        inside_pc = pc[inside_mask]

        if inside_pc.shape[0] < num_samples:
            raise ValueError(f"Not enough points inside the bounding box to sample: {inside_pc.shape[0]}")

        # Initialize variables for FPS
        sampled_indices = torch.zeros(num_samples, dtype=torch.long, device=pc.device)
        distances = torch.full((inside_pc.shape[0],), float('inf'), device=pc.device)

        # Start with a random point
        sampled_indices[0] = torch.randint(0, inside_pc.shape[0], (1,), device=pc.device)

        for i in range(1, num_samples):
            # Compute distances from all points to the nearest sampled point
            dist = torch.norm(inside_pc[:, :3] - inside_pc[sampled_indices[i - 1], :3], dim=1)
            distances = torch.min(distances, dist)

            # Select the farthest point
            sampled_indices[i] = torch.argmax(distances)

        # Gather the sampled points
        sampled_pc = inside_pc[sampled_indices]

        return sampled_pc
    
    def RPS_with_bounding_box(self, pc, num_samples, bounding_box):
      
        # Extract the min and max corners of the bounding box
        min_corner = bounding_box[0]
        max_corner = bounding_box[1]
        
        pc_xyz = pc[:, :3]
        # print(f'pc_xyz.shape: {pc_xyz.shape}')
        # Filter points to only include those within the bounding box
        inside_mask = (pc_xyz >= min_corner) & (pc_xyz <= max_corner)
        inside_mask = inside_mask[:, 0] & inside_mask[:, 1] & inside_mask[:, 2]
        inside_pc = pc[inside_mask]

        rand_idx = np.random.choice(inside_pc.shape[0], num_samples, replace=False)
        return inside_pc[rand_idx]
    
# --------------------------------------------------------------------
    def save_weights(self, savedir: str):
        torch.save(self._actor.state_dict(),
                   os.path.join(savedir, f'ppi_{self.weight_name}.pth.tar'))

    def normalize_z(self, data, mean, std):
        return (data - mean) / std

    def unnormalize_z(self, data, mean, std):
        return data * std + mean
    
    def _normalize_quat(self, x):
        return x / x.square().sum(dim=1).sqrt().unsqueeze(-1)

    def _normalize_revolute_joints(self, x):
        # normalize joint angles
        # input ranges from -pi to pi
        # out ranges from 0 to 1
        return (x + np.pi) / (2 * np.pi)

    def _unnormalize_revolute_joints(self, x):
        # map input with range 0 to 1 to -pi to pi
        x = (x - 0.5) * 2.0 * np.pi
        x = torch.clamp(x, -np.pi, np.pi)
        return x

    def _normalize_gripper_joints(self, x):
        gripper_min = 0
        gripper_max = 0.04
        # normalize gripper joint angles between 0 and 1, the input ranges from 0 to 0.04
        return ((x - gripper_min) / (gripper_max - gripper_min))

    def _unnormalize_gripper_joints(self, x):
        gripper_min = 0
        gripper_max = 0.04
        
        x = x * (gripper_max - gripper_min) + gripper_min
        x = torch.clamp(x, gripper_min, gripper_max)
        return torch.unsqueeze(x, dim=0)