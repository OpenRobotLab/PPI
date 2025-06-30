from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy
import time
import pytorch3d.ops as torch3d_ops

from ppi.model.common.normalizer import LinearNormalizer
from ppi.policy.base_policy import BasePolicy
from ppi.model.diffusion.mask_generator import LowdimMaskGenerator
from ppi.common.pytorch_util import dict_apply
from ppi.common.model_util import print_params

from ppi.model.vision.observation_encoder import ObservationEncoder
from ppi.model.diffusion.diffuser_actor_ppi_real import DiffusionHeadPPIReal
from ppi.model.diffusion.diffuser_actor_ppi_real_simple import DiffusionHeadPPIRealSimple


import os
from pdb import set_trace

class PPI(BasePolicy):
    def __init__(self, 
            noise_scheduler_position: DDPMScheduler, 
            noise_scheduler_rotation: DDPMScheduler, 
            horizon_keyframe,
            horizon_continuous,
            n_action_steps, 
            n_obs_steps, 
            num_inference_steps=None, 
            encoder_output_dim=288, 
            use_lang=False,
            pointcloud_encoder_cfg=None, 
            what_condition='none',
            predict_point_flow=True,
            **kwargs):
        super().__init__()

        self.what_condition = what_condition
        self.predict_point_flow = predict_point_flow

        obs_encoder = ObservationEncoder(
                                out_channel=encoder_output_dim,
                                pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                use_lang=use_lang,
                                use_initial_pointflow=predict_point_flow
                                )

        self.position_noise_scheduler = noise_scheduler_position
        self.rotation_noise_scheduler = noise_scheduler_rotation

        self.what_condition=what_condition

        if what_condition =='ppi_real':
            model =  DiffusionHeadPPIReal(
                embedding_dim=encoder_output_dim, 
                num_attn_heads=8,
                use_instruction=True,
                rotation_parametrization='quat',
                nhist=1,
                lang_enhanced=False,
                horizon_keyframe = horizon_keyframe,
                horizon_continuous = horizon_continuous
                )
        if what_condition =='ppi_real_simple':
            model =  DiffusionHeadPPIRealSimple(
                embedding_dim=encoder_output_dim, 
                num_attn_heads=8,
                use_instruction=True,
                rotation_parametrization='quat',
                nhist=1,
                lang_enhanced=False,
                horizon_keyframe = horizon_keyframe,
                horizon_continuous = horizon_continuous
                )  
        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler_position
        
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=True
        )
        
        self.normalizer = LinearNormalizer()
        self.horizon_keyframe = horizon_keyframe
        self.horizon_continuous = horizon_continuous
        self.horizon = horizon_keyframe + horizon_continuous
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print_params(self)
        
    # ========= inference  ============
    def conditional_sample_diffuser_actor(self, condition_data, condition_mask, fixed_inputs,
                                          condition_object_pose=None,condition_mask_object_pose=None,
                                          condition_point_flow=None,condition_mask_point_flow=None):
        # set_trace()
        self.position_noise_scheduler.set_timesteps(self.num_inference_steps)
        self.rotation_noise_scheduler.set_timesteps(self.num_inference_steps)
        model = self.model
        noise = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device
        )         
        noise_left = noise[..., :7]
        noise_right = noise[..., 7:14]
        noise_t = torch.ones(
            (len(condition_data),), device=condition_data.device
        ).long().mul(self.position_noise_scheduler.timesteps[0])
        noise_pos_left = self.position_noise_scheduler.add_noise(
            condition_data[..., :3], noise[..., :3], noise_t
        )
        noise_rot_left = self.rotation_noise_scheduler.add_noise(
            condition_data[..., 3:7], noise[..., 3:7], noise_t
        )
        noise_pos_right = self.position_noise_scheduler.add_noise(
            condition_data[..., 7:10], noise[..., 7:10], noise_t
        )
        noise_rot_right = self.rotation_noise_scheduler.add_noise(
            condition_data[..., 10:14], noise[..., 10:14], noise_t
        )
                   
        
        noisy_condition_data_left = torch.cat(
            (noise_pos_left, noise_rot_left
             ), -1)
        
        noisy_condition_data_right = torch.cat(
            (noise_pos_right, noise_rot_right
             ), -1)
        condition_mask_left = condition_mask[..., :7]
        condition_mask_right = condition_mask[..., 7:14]
        trajectory_left = torch.where(
            condition_mask_left, noisy_condition_data_left, noise_left
        )
        trajectory_right = torch.where(
            condition_mask_right, noisy_condition_data_right, noise_right
        )
        
        timesteps = self.position_noise_scheduler.timesteps
        for t in timesteps:
            out_left, out_right, out_point_flow = model(
                trajectory_left,trajectory_right,
                t * torch.ones(len(trajectory_left)).to(trajectory_left.device).long(),
                fixed_inputs
            )

            out_left = out_left[-1]  # keep only last layer's output
            out_right = out_right[-1]
            pos_left = self.position_noise_scheduler.step(
                out_left[..., :3], t, trajectory_left[..., :3]
            ).prev_sample
            rot_left = self.rotation_noise_scheduler.step(
                out_left[..., 3:7], t, trajectory_left[..., 3:7]
            ).prev_sample
            pos_right = self.position_noise_scheduler.step(
                out_right[..., :3], t, trajectory_right[..., :3]
            ).prev_sample
            rot_right = self.rotation_noise_scheduler.step(
                out_right[..., 3:7], t, trajectory_right[..., 3:7]
            ).prev_sample

            trajectory_left = torch.cat((pos_left, rot_left), -1)
            trajectory_right = torch.cat((pos_right, rot_right), -1)
            trajectory = torch.cat((pos_left, rot_left,
                                        pos_right, rot_right,
                                        out_left[..., 7:8], out_right[..., 7:8]), -1)

        if self.predict_point_flow:
            return trajectory, out_point_flow[-1]
        else:
            return trajectory
        
    
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        this_n_point_cloud = nobs['point_cloud']
        
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon 
        Da = self.action_dim 
        To = self.n_obs_steps 
        # set_trace()
        # build input
        device = self.device
        dtype = self.dtype
        
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))

        

        (context_coord, context_feat, lang_feat, state_feat, pn_coord, pn_feat, pointflow_feat, pointflow_coords) = self.obs_encoder(this_nobs)
        lang_feat = lang_feat.reshape(B, To, -1)
        state_feat = state_feat.reshape(B, To, -1)
        fixed_inputs = (context_coord, context_feat, lang_feat, state_feat, pn_coord, pn_feat, pointflow_feat, pointflow_coords)
        
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        fps = this_nobs["initial_point_flow"].shape[1]
        cond_point_flow = torch.zeros(size=(B, self.horizon_keyframe*fps, 3), device=device, dtype=dtype)
        cond_mask_point_flow = torch.zeros_like(cond_point_flow, dtype=torch.bool)
        # run sampling
        nsample, npoint_flow = self.conditional_sample_diffuser_actor(cond_data, cond_mask, fixed_inputs, 
                                                                                    condition_point_flow=cond_point_flow,
                                                                                    condition_mask_point_flow=cond_mask_point_flow)
        naction_pred = nsample[...,:Da]
        npoint_flow_pred = npoint_flow
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        point_flow_pred = self.normalizer['point_flow'].unnormalize(npoint_flow_pred)
        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        # get prediction


        result = {
            'action': action,
            'action_pred': action_pred,
            'point_flow_pred': point_flow_pred
        }

        return result
            

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def forward(self, batch):
        # normalize input
        # set_trace()
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        if self.predict_point_flow:
            npoint_flow = self.normalizer['point_flow'].normalize(batch['point_flow'])
        nobs['point_cloud'] = nobs['point_cloud'][..., :3]

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        trajectory = nactions 
        cond_data = trajectory
        this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        if self.predict_point_flow:
            point_flow_trajectory = npoint_flow[:,-self.horizon_keyframe:,:,:]
            point_flow_trajectory = point_flow_trajectory.reshape(batch_size, -1, 3)
            point_flow_cond_data = point_flow_trajectory
            (context_coord, context_feat, lang_feat, state_feat, pn_coord, pn_feat, pointflow_feat, pointflow_coords) = self.obs_encoder(this_nobs)
            lang_feat = lang_feat.reshape(batch_size, self.n_obs_steps, -1)
            state_feat = state_feat.reshape(batch_size, self.n_obs_steps, -1)
            fixed_inputs = (context_coord, context_feat, lang_feat, state_feat, pn_coord, pn_feat, pointflow_feat, pointflow_coords)

        condition_mask = self.mask_generator(trajectory.shape)

        noise = torch.randn(trajectory.shape, device=trajectory.device)

        
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        gt_openess_left = trajectory[..., 14:15]
        gt_openess_right = trajectory[..., 15:16]
        # set_trace()
        pos_left = self.position_noise_scheduler.add_noise(
            trajectory[..., :3], noise[..., :3], timesteps
        )
        rot_left = self.rotation_noise_scheduler.add_noise(
            trajectory[..., 3:7], noise[..., 3:7], timesteps
        )
        pos_right = self.position_noise_scheduler.add_noise(
            trajectory[..., 7:10], noise[..., 7:10], timesteps
        )
        rot_right = self.rotation_noise_scheduler.add_noise(
            trajectory[..., 10:14], noise[..., 10:14], timesteps
        )

        noisy_trajectory = torch.cat((pos_left, rot_left, pos_right, rot_right,gt_openess_left,gt_openess_right), -1)
        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        noisy_trajectory_left = noisy_trajectory[..., :7]
        noisy_trajectory_right = noisy_trajectory[..., 7:14]
        # Predict the noise residual
        if self.predict_point_flow:
            pred_left, pred_right, pred_point_flow = self.model(
                noisy_trajectory_left, noisy_trajectory_right,
                timesteps,
                fixed_inputs
            )
        
        total_loss = 0
        for layer_pred_left in pred_left:
            trans_left = layer_pred_left[..., :3]
            rot_left = layer_pred_left[..., 3:7]
            loss_left = (
                30 * F.l1_loss(trans_left, noise[..., :3], reduction='mean')
                + 10 * F.l1_loss(rot_left, noise[..., 3:7], reduction='mean')
            )
            if torch.numel(gt_openess_left) > 0:
                openess_left = layer_pred_left[..., 7:8]
                loss_left += 30*F.l1_loss(openess_left, gt_openess_left, reduction='mean')
            total_loss = total_loss + loss_left
            
        for layer_pred_right in pred_right:
            trans_right = layer_pred_right[..., :3]
            rot_right = layer_pred_right[..., 3:7]
            loss_right = (
                30 * F.l1_loss(trans_right, noise[...,7:10], reduction='mean')
                + 10 * F.l1_loss(rot_right, noise[..., 10:14], reduction='mean')
            )
            if torch.numel(gt_openess_right) > 0:
                openess_right = layer_pred_right[..., 7:8]
                loss_right += 30*F.l1_loss(openess_right, gt_openess_right, reduction='mean')
            total_loss = total_loss + loss_right
        loss_dict = {
                'action_loss': loss_left.item()+loss_right.item(),
            }
        
        if self.predict_point_flow:
            for layer_pred_point_flow in pred_point_flow:
                trans_point_flow = layer_pred_point_flow[..., :3]
                loss_point_flow = (
                    600 * F.l1_loss(trans_point_flow, point_flow_trajectory[...,:3], reduction='mean')
                )
                total_loss = total_loss + loss_point_flow
            loss_dict["point_flow_loss"] = loss_point_flow.item()
        loss_dict["bc_loss"] = total_loss.item()
        return total_loss, loss_dict

