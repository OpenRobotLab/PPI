import numpy as np
from scipy.spatial.transform import Rotation as R

def xyzquat2xyzypr(pose):
    x, y, z, qw, qx, qy, qz = pose
    quat = np.array([qx, qy, qz, qw])
    quat = quat / np.linalg.norm(quat)
    r = R.from_quat(quat)
    yaw, pitch, roll = r.as_euler('ZYX', degrees=False)
    return [x, y, z, yaw, pitch, roll]

def xyzypr2xyzquat(pose):
    x, y, z, yaw, pitch, roll = pose
    r = R.from_euler('ZYX', [yaw, pitch, roll])
    qx, qy, qz, qw  = r.as_quat()
    return [x, y, z, qw, qx, qy, qz]

def xyzypr2T(pose):
    x, y, z, yaw, pitch, roll = pose
    r = R.from_euler('ZYX', [yaw, pitch, roll])
    transform = np.eye(4)
    transform[:3, :3] = r.as_matrix()
    transform[:3, 3] = np.array([x, y, z])
    return transform

def T2xyzypr(T):
    x, y, z = T[:3, 3]
    r = R.from_matrix(T[:3, :3])
    yaw, pitch, roll = r.as_euler('ZYX', degrees=False)
    return [x, y, z, yaw, pitch, roll]

def xyzquat2T(pose):
    x, y, z, qw, qx, qy, qz = pose
    quat = np.array([qx, qy, qz, qw])
    quat = quat / np.linalg.norm(quat)
    r = R.from_quat(quat)
    transform = np.eye(4)
    transform[:3, :3] = r.as_matrix()
    transform[:3, 3] = np.array([x, y, z])
    return transform