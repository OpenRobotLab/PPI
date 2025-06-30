import roboticstoolbox as rtb
import numpy as np
from pathlib import Path
from pdb import set_trace
import pickle as pkl
# In real world, GELLO is used to control the robot.
# You can choose other controllers for tele-operation.
from gello.agents.gello_agent import GelloAgent
from frankx import Kinematics, Affine
import time
from utils.transform_utils import xyzypr2T, T2xyzypr

PI = np.pi

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

class TeleController():
    def __init__(self, control_type="pose", env=None):
        self.control_type = control_type
        self.env = env
        self.gello_port_l = '/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT9HDF4L-if00-port0' # left
        self.gello_port_r = '/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT9HDF4R-if00-port0' # right
        self.agent_l = GelloAgent(port=self.gello_port_l)
        self.agent_r = GelloAgent(port=self.gello_port_r)

        # set_trace()
        init_time = 3
        print(f'Initialize the gellos in {init_time} seconds. Please keep the gellos stable.')
        time.sleep(init_time)
        self.agent_l._robot.set_torque_mode(False)
        self.agent_r._robot.set_torque_mode(False)
        self.iap_l_T, self.iap_r_T, self.iap_l_T_inv, self.iap_r_T_inv,\
             self.ex_iap_l_T, self.ex_iap_r_T, self.ex_iap_l_T_inv, self.ex_iap_r_T_inv = self.get_init_action_pose()
        

    def update_init_action_pose(self):
        self.iap_l_T, self.iap_r_T, self.iap_l_T_inv, self.iap_r_T_inv,\
             self.ex_iap_l_T, self.ex_iap_r_T, self.ex_iap_l_T_inv, self.ex_iap_r_T_inv = self.get_init_action_pose()

    def get_init_action_pose(self):
        obs = None
        action_joint_l = self.agent_l.act(obs)
        action_joint_r = self.agent_r.act(obs)

        init_action_pose_l = Affine(Kinematics.forward(action_joint_l[:7] + [0, 0, 0, 0, 0, 0, -PI/4])).vector() + np.array([0, 0, -0.1034, 0, 0, 0])
        init_action_pose_r = Affine(Kinematics.forward(action_joint_r[:7] + [0, 0, 0, 0, 0, 0, -PI/4])).vector() + np.array([0, 0, -0.1034, 0, 0, 0])
        
        iap_l_T = xyzypr2T(init_action_pose_l)
        iap_r_T = xyzypr2T(init_action_pose_r)

        iap_l_T_inv = np.linalg.inv(iap_l_T)
        iap_r_T_inv = np.linalg.inv(iap_r_T)

        ex_iap_l_T = np.eye(4)
        ex_iap_l_T[:3, :3] = iap_l_T[:3, :3]
        ex_iap_r_T = np.eye(4)
        ex_iap_r_T[:3, :3] = iap_r_T[:3, :3]

        ex_iap_l_T_inv = np.linalg.inv(ex_iap_l_T)
        ex_iap_r_T_inv = np.linalg.inv(ex_iap_r_T)
        
        return iap_l_T, iap_r_T, iap_l_T_inv, iap_r_T_inv, ex_iap_l_T, ex_iap_r_T, ex_iap_l_T_inv, ex_iap_r_T_inv

    def preprocess(self, gello_action_pose_l, gello_action_pose_r):
        gap_l_T = xyzypr2T(gello_action_pose_l)
        gap_r_T = xyzypr2T(gello_action_pose_r)

        action_pose_l_T = HOME_POSE_L @ self.ex_iap_l_T @ self.iap_l_T_inv @ gap_l_T @ self.ex_iap_l_T_inv
        action_pose_r_T = HOME_POSE_R @ self.ex_iap_r_T @ self.iap_r_T_inv @ gap_r_T @ self.ex_iap_r_T_inv

        action_pose_l = T2xyzypr(action_pose_l_T)
        action_pose_r = T2xyzypr(action_pose_r_T)

        return action_pose_l, action_pose_r

    def __call__(self, obs):
        action_joint_l = self.agent_l.act(obs)
        action_joint_r = self.agent_r.act(obs)
        gello_action_pose_l = Affine(Kinematics.forward(action_joint_l[:7] + [0, 0, 0, 0, 0, 0, -PI/4])).vector() + np.array([0, 0, -0.1034, 0, 0, 0])
        gello_action_pose_r = Affine(Kinematics.forward(action_joint_r[:7] + [0, 0, 0, 0, 0, 0, -PI/4])).vector() + np.array([0, 0, -0.1034, 0, 0, 0])

        action_pose_l, action_pose_r = self.preprocess(gello_action_pose_l, gello_action_pose_r)

        if action_joint_l[7] > 0.5:
            gripper_l = 0
        else:
            gripper_l = 1

        if action_joint_r[7] > 0.5:
            gripper_r = 0
        else:
            gripper_r = 1

        return action_pose_l, action_pose_r, gripper_l, gripper_r