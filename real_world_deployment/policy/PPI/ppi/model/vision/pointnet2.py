import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import ball_query, sample_farthest_points
from time import time
import numpy as np
from typing import Optional, Tuple, Union, List
from pdb import set_trace
def test_sample_farthest_points():
    B, N, C = 2, 100, 3
    xyz = torch.randn(B, N, C)
    
    old_idx = old_farthest_point_sample(xyz, 10)
    old_idx = old_idx.sort()[0]
    old_sampled_xyz = index_points(xyz, old_idx)
    
    sampled_xyz, idx = sample_farthest_points(xyz, K = 10, random_start_point=False)
    for b in range(B):
        sampled_xyz[b] = sampled_xyz[b][idx.sort()[1][b]]
    idx = idx.sort()[0]
    
    print("new sampled xyz: ", sampled_xyz)
    print("old sampled xyz: ", old_sampled_xyz)
    print("new idx: ", idx)
    print("old idx: ", old_idx)

def test_ball_query():   
    B, N, C = 2, 100, 3
    M = 10
    xyz = torch.randn(B, N, C)
    new_xyz = torch.randn(B, M, C)
    radius = 2.0
    nsample = 16
    
    _, new_idx, new_grouped_xyz = ball_query(new_xyz, xyz, K=nsample, radius=radius, return_nn=True)
    
    old_idx = old_query_ball_point(radius, nsample, xyz, new_xyz)
    old_grouped_xyz = index_points(xyz, old_idx)
    
    print("new idx: ", new_idx)
    print("old idx: ", old_idx)
    print('')

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def old_farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(1, npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def old_query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    new_xyz, fps_idx = sample_farthest_points(xyz, K = npoint) # [B, npoint, C]
    _, idx, grouped_xyz = ball_query(new_xyz, xyz, K=nsample, radius=radius, return_nn=True) # [B, npoint, nsample]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, bn=True):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if self.bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        if nsample is not None:
            self.max_pool = nn.MaxPool2d((nsample, 1))
        else:
            self.max_pool = None
        self.group_all = group_all
        self.identity = nn.Identity() # hack to get new_xyz

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
            
        # set_trace()

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                new_points =  F.relu(bn(conv(new_points)))
            else:
                new_points = F.relu(conv(new_points)) # [B, C+D, nsample, npoint]

        if self.max_pool is not None:
            new_points = self.max_pool(new_points)[:, :, 0]
        else:
            new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        new_xyz = self.identity(new_xyz)
        return new_xyz, new_points


class PointNet2DenseEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=256, use_bn=True, npoint1=3072, npoint2=1024):
        super(PointNet2DenseEncoder, self).__init__()
        
        self.sa1 = PointNetSetAbstraction(npoint=npoint1, radius=0.04, nsample=32, in_channel=in_channels, mlp=[64, 64, 128], group_all=False,bn=True)
        self.sa2 = PointNetSetAbstraction(npoint=npoint2, radius=0.08, nsample=64, in_channel=128+3, mlp=[128, 128, 288], group_all=False,bn=True)
        
        # copy variables
        self.in_channels = in_channels
        
    def forward(self, xyz):
        B, D, N = xyz.size()
        assert D == self.in_channels
        if D > 3:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) 
        return (l2_xyz,l2_points)
    

if __name__ == '__main__':
    # test_sample_farthest_points()
    test_ball_query()