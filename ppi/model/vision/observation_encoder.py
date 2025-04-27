import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
import os
from ppi.model.vision.pointnet2 import PointNet2DenseEncoder
from pdb import set_trace
def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules

class ObservationEncoder(nn.Module):
    def __init__(self, 
                 out_channel=288,
                 state_mlp_size=(128, 288), state_mlp_activation_fn=nn.ReLU,
                 lang_mlp_size=(288, 288), lang_mlp_activation_fn=nn.ReLU,
                 pcd_mlp_size=(288,288), pcd_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_lang=False,
                 use_initial_pointflow=True,
                 scene_pcd_num=6144,
                 ):
        super().__init__()
        self.scene_pcd_num = scene_pcd_num
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.lang_key = 'lang'
        self.out_channel = out_channel
        
        self.state_shape = 16   # dual arm ee pose
        self.lang_shape = 1024  
        self.use_initial_pointflow=use_initial_pointflow
        self.pcd_mlp_size = pcd_mlp_size
        self.pointnet_encoder = PointNet2DenseEncoder(**pointcloud_encoder_cfg)
        

        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape, output_dim, net_arch, state_mlp_activation_fn))
        self.pcd_mlp = nn.Sequential(*create_mlp(384, pcd_mlp_size[-1], pcd_mlp_size[:-1], pcd_mlp_activation_fn))
        if self.use_initial_pointflow:
            self.point_flow_mlp = nn.Sequential(*create_mlp(3, pcd_mlp_size[-1], pcd_mlp_size[:-1], pcd_mlp_activation_fn))
        if use_lang:
            if len(lang_mlp_size) == 0:
                raise RuntimeError(f"Language mlp size is empty")
            elif len(lang_mlp_size) == 1:
                lang_net_arch = []
            else:
                lang_net_arch = lang_mlp_size[:-1]
            lang_dim = lang_mlp_size[-1]
            self.lang_mlp = nn.Sequential(*create_mlp(self.lang_shape, lang_dim, lang_net_arch, lang_mlp_activation_fn))


    def forward(self, observations: Dict) -> torch.Tensor:
        # set_trace()
        points = observations[self.point_cloud_key]
        dino_feature = observations["dino_feature"]
        if self.use_initial_pointflow:
            initial_point_flow = observations["initial_point_flow"]
        ptc_wth_feature = torch.cat((points,dino_feature),dim=2)

        ptc_wth_feature = torch.transpose(ptc_wth_feature, 1, 2)
        (sampled_pcd_coord, sampled_pcd_feat) = self.pointnet_encoder(ptc_wth_feature)    # B * out_channel
        sampled_pcd_coord = torch.transpose(sampled_pcd_coord,1,2)
        sampled_pcd_feat = torch.transpose(sampled_pcd_feat,1,2)
        
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)
        lang = observations[self.lang_key]
        lang_feat = self.lang_mlp(lang)
        dino_feature = dino_feature.reshape(-1,384)
        pcd_feat = self.pcd_mlp(dino_feature).reshape(-1,self.scene_pcd_num,self.pcd_mlp_size[-1])
        
        if self.use_initial_pointflow:
            fps = initial_point_flow.shape[1]
            initial_point_flow = initial_point_flow.reshape(-1,3)
            point_flow_feat = self.point_flow_mlp(initial_point_flow).reshape(-1,fps,self.pcd_mlp_size[-1])
            return (points, pcd_feat, lang_feat, state_feat, sampled_pcd_coord, sampled_pcd_feat, point_flow_feat, initial_point_flow.reshape(-1,fps,3))
        return (points, pcd_feat, lang_feat, state_feat, sampled_pcd_coord, sampled_pcd_feat)


    def output_shape(self):
        return self.out_channel