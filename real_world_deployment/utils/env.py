import roboticstoolbox as rtb
from frankx import Robot, LinearMotion, JointMotion, WaypointMotion, Affine, Waypoint, ImpedanceMotion, InvalidOperationException
import numpy as np
from pathlib import Path
import time
from pdb import set_trace
import pickle as pkl
from threading import Thread, Condition
import threading
import pyrealsense2 as rs
import cv2
from .realsense import MultiRealSenseCamera
import json
import torch
 

PI = np.pi
HOME_JOINTS_l = [- PI / 6, - PI / 4, 0, -3 * PI / 4, 0, PI / 2, PI / 4 - PI / 6]
HOME_JOINTS_r = [PI / 6, - PI / 4, 0, -3 * PI / 4, 0, PI / 2, PI / 4 + PI / 6]
HOME_POSE = Affine(0.3069, 0., 0.4867, 0, 0, 0.)
HOME_POSE_ARRAY = [0.3069, 0., 0.4867, 0, 0, 0.]

class Rate:
    def __init__(self, rate: float):
        self.last = time.time()
        self.rate = rate

    def sleep(self) -> None:
        while self.last + 1.0 / self.rate > time.time():
            time.sleep(0.0001)
        self.last = time.time()

class PPIRealEnv():
    def __init__(self, 
                 robot_ip1="172.16.0.2", 
                 robot_ip2="172.16.1.2", 
                 control_freq=100, 
                 control_type="pose",
                 left_extr_path=None,
                 right_extr_path=None,
                 head_extr_path=None,
                 if_tele=False,
                 trans_stiffness=300.0,
                 rot_stiffness=30.0):
        """
        Initializes the environment with two robots, setting their IP addresses, control frequency, and control type.
        Args:
            robot_ip1 (str): IP address of the first robot. Default is "172.16.0.2".
            robot_ip2 (str): IP address of the second robot. Default is "172.16.1.2".
            control_freq (int): Control frequency for the robots. Default is 100.
            control_type (str): Type of control to be used. Default is "pose".
        Attributes:
            robot1 (Robot): Instance of the first robot.
            robot2 (Robot): Instance of the second robot.
            control_freq (int): Control frequency for the robots.
            control_type (str): Type of control to be used.
        """
        
        self.robot1 = Robot(robot_ip1)
        self.robot2 = Robot(robot_ip2)
        self.control_freq = control_freq

        self.robot1.set_default_behavior()
        self.robot1.recover_from_errors()
        self.robot1.velocity_rel = 1.0 / 6
        self.robot1.acceleration_rel = 0.6 / 6
        self.robot1.jerk_rel = 0.01 / 6

        self.robot2.set_default_behavior()
        self.robot2.recover_from_errors()
        self.robot2.velocity_rel = 1.0 / 6
        self.robot2.acceleration_rel = 0.6 / 6
        self.robot2.jerk_rel = 0.01 / 6

        self.gripper1 = self.robot1.get_gripper()
        self.gripper2 = self.robot2.get_gripper()
        self.gripper1.gripper_speed = 0.2
        self.gripper2.gripper_speed = 0.2

        threading1 = self.robot1.move_async(JointMotion(HOME_JOINTS_l))
        threading2 = self.robot2.move_async(JointMotion(HOME_JOINTS_r))
        # threading3 = Thread(target=self.gripper1.open)
        threading3 = Thread(target=self.gripper1.move, args=(0.07,))
        threading3.start()
        # threading4 = Thread(target=self.gripper2.open)
        threading4 = Thread(target=self.gripper2.move, args=(0.07,))
        threading4.start()

        threading1.join()
        threading2.join()
        threading3.join()
        threading4.join()
        
        self.gripper1_open = 1
        self.gripper2_open = 1

        self.control_type = control_type

        self.trans_stiffness = trans_stiffness
        self.rot_stiffness = rot_stiffness

        # realsense initialization
        # TODO: modify the realsense camera id according to your setup
        camera_cfg = {'340422073383': 'left',
                    '342222072145': 'right',
                    '327122070263': 'head'}
        self.multi_camera = MultiRealSenseCamera(fps=30, camera_cfg=camera_cfg)

        self.left_extr_path = left_extr_path
        self.right_extr_path = right_extr_path
        self.head_extr_path = head_extr_path
        self.panda = rtb.models.Panda()

        self.init_extrinsics()

        self.is_first_frame = True

    def get_robot_state(self):
        try:
            robot1_current_pose = self.robot1.current_pose(read_once=True)
        except InvalidOperationException:
            robot1_current_pose = self.robot1.current_pose(read_once=False)


        try:
            robot2_current_pose = self.robot2.current_pose(read_once=True)
        except InvalidOperationException:
            robot2_current_pose = self.robot2.current_pose(read_once=False)

        # q_pose
        try:
            robot1_current_joints = self.robot1.current_joint_positions(read_once=True)
        except InvalidOperationException:
            robot1_current_joints = self.robot1.current_joint_positions(read_once=False)

        try:
            robot2_current_joints = self.robot2.current_joint_positions(read_once=True)
        except InvalidOperationException:
            robot2_current_joints = self.robot2.current_joint_positions(read_once=False)

        # gripper_open
        gripper1_open = self.gripper1_open
        gripper2_open = self.gripper2_open

        return robot1_current_pose.translation().tolist() + robot1_current_pose.quaternion(), \
            robot2_current_pose.translation().tolist() + robot2_current_pose.quaternion(), \
            robot1_current_joints, robot2_current_joints, gripper1_open, gripper2_open

    def get_robot_pose_matrix(self):
        try:
            robot1_current_joints = self.robot1.current_joint_positions(read_once=True)
        except InvalidOperationException:
            robot1_current_joints = self.robot1.current_joint_positions(read_once=False)

        try:
            robot2_current_joints = self.robot2.current_joint_positions(read_once=True)
        except InvalidOperationException:
            robot2_current_joints = self.robot2.current_joint_positions(read_once=False)

        robot1_pose_matrix = self.panda.fkine(robot1_current_joints, end="panda_hand").A
        robot2_pose_matrix = self.panda.fkine(robot2_current_joints, end="panda_hand").A

        return robot1_pose_matrix, robot2_pose_matrix
    
    def get_robot_joints(self):
        try:
            robot1_current_joints = self.robot1.current_joint_positions(read_once=True)
        except InvalidOperationException:
            robot1_current_joints = self.robot1.current_joint_positions(read_once=False)

        try:
            robot2_current_joints = self.robot2.current_joint_positions(read_once=True)
        except InvalidOperationException:
            robot2_current_joints = self.robot2.current_joint_positions(read_once=False)

        return robot1_current_joints, robot2_current_joints
    
    def get_camera_data(self, image_size, x10000_depth=False):
        done = False
        rgb_intr = self.multi_camera.get_intrinsic_color()
        extr = self.get_extrinsics()

        color_image, depth_image = self.multi_camera.undistorted_rgbd(x10000_depth=x10000_depth)

        for camera in color_image.keys():
            cv2.imshow(f"Color Image - {camera}", cv2.cvtColor(color_image[camera], cv2.COLOR_RGB2BGR))

        op = cv2.waitKey(1) & 0xFF

        if op == ord('s'):
            self.flag_start = True
            print(f"==== Start Saving ====")

        if op == ord('k'):
            is_keyframe = True
            print("==== Add Keyframe ====")
        else:
            is_keyframe = False
        
        if op == ord('c'):
            done = True
            print("==== Done ====")

        return color_image, depth_image, rgb_intr, extr, is_keyframe, done
    
    

    def get_extrinsics(self):
        extr = {}
        T_leftwrist_2_leftbase, T_rightwrist_2_rightbase = self.get_robot_pose_matrix()
        extr['left'] = self.T_leftbase_2_world @ T_leftwrist_2_leftbase @ self.T_leftcamera_2_leftwrist
        extr['right'] =  self.T_rightbase_2_world @ T_rightwrist_2_rightbase @ self.T_rightcamera_2_rightwrist
        extr['head'] = self.T_leftbase_2_world @ self.T_headcamera_2_leftbase
        return extr

    def init_extrinsics(self):
        self.T_leftcamera_2_leftwrist = self.get_transformation_from_json(self.left_extr_path)
        self.T_rightcamera_2_rightwrist = self.get_transformation_from_json(self.right_extr_path)
        self.T_headcamera_2_leftbase = self.get_transformation_from_json(self.head_extr_path)

        self.T_leftbase_2_world = np.array([[1, 0, 0, 0], 
                                            [0, 1, 0, 0.905/2], 
                                            [0, 0, 1, 0], 
                                            [0, 0, 0, 1]])

        self.T_rightbase_2_world = np.array([[1, 0, 0, 0], 
                                            [0, 1, 0, -0.905/2], 
                                            [0, 0, 1, 0], 
                                            [0, 0, 0, 1]])
        

    def get_transformation_from_json(self, path):
        with open(path, 'r') as f:
            data = json;.load(f)
        transformation = np.eye(4)
        transformation[:3, :3] = np.array(data['rotation_matrix'])
        transformation[:3, 3] = np.array([data['translation']['x'], data['translation']['y'], data['translation']['z']])
        return transformation


    def start(self):
        self.flag_start = False
        self.first_pred = True

        if self.control_type == "pose":
            robot1_current_pose, robot2_current_pose, _, _, _, _ = self.get_robot_state()
            # set initial pose
            self.robot1_tgt_pose = Affine(*robot1_current_pose)
            self.robot1_impedance_motion = ImpedanceMotion(self.trans_stiffness, self.rot_stiffness)
            self.robot1_motion_thread = self.robot1.move_async(self.robot1_impedance_motion)

            self.robot2_tgt_pose = Affine(*robot2_current_pose)
            self.robot2_impedance_motion = ImpedanceMotion(self.trans_stiffness, self.rot_stiffness)
            self.robot2_motion_thread = self.robot2.move_async(self.robot2_impedance_motion)

            return self.robot1_motion_thread, self.robot2_motion_thread
        
        elif self.control_type == "joint":
            robot1_current_joints, robot2_current_joints = self.get_robot_joints()

            self.robot1_tgt_joints = robot1_current_joints
            self.robot1_motion = JointMotion(self.robot1_tgt_joints)
            self.robot1_WaypointMotionQPose = WaypointMotionQPose(self.robot1)
            self.robot1_motion_thread = self.robot1_WaypointMotionQPose.move_async(self.robot1_motion)

            self.robot2_tgt_joints = robot2_current_joints
            self.robot2_motion = JointMotion(self.robot2_tgt_joints)
            self.robot2_WaypointMotionQPose = WaypointMotionQPose(self.robot2)
            self.robot2_motion_thread = self.robot2_WaypointMotionQPose.move_async(self.robot2_motion)

            return self.robot1_motion_thread, self.robot2_motion_thread

    def step(self, robot1_action, robot2_action, gripper1_open, gripper2_open):
        
        if self.control_type == "pose":
            self.robot1_impedance_motion.target = Affine(*robot1_action)
            self.robot2_impedance_motion.target = Affine(*robot2_action)
            
        elif self.control_type == "joint":
            self.robot1_WaypointMotionQPose.set_next_waypoint(JointMotion(robot1_action))
            self.robot2_WaypointMotionQPose.set_next_waypoint(JointMotion(robot2_action))
        
        if gripper1_open != self.gripper1_open:
            self.gripper1_open = gripper1_open
            if gripper1_open == 1:
                # gripper1_thread = Thread(target=self.gripper1.open) # gripper.move(0.05)
                gripper1_thread = Thread(target=self.gripper1.move, args=(0.07,)) # gripper.move(0.05)
            elif gripper1_open == 0:
                gripper1_thread = Thread(target=self.gripper1.clamp)
            gripper1_thread.start()

        if gripper2_open != self.gripper2_open:
            self.gripper2_open = gripper2_open
            if gripper2_open == 1:
                # gripper2_thread = Thread(target=self.gripper2.open)
                gripper2_thread = Thread(target=self.gripper2.move, args=(0.07,))
            elif gripper2_open == 0:
                gripper2_thread = Thread(target=self.gripper2.clamp)
            gripper2_thread.start()

        return None
    
    def shutdown(self):
        # wait for the last motion to finish
        time.sleep(0.5)

        # stop the camera
        # for pipeline in self.multi_camera.pipelines:
        #     pipeline.stop()
        cv2.destroyAllWindows()

        # stop the robot motion
        if self.control_type == "pose":
            self.robot1_impedance_motion.finish()
            self.robot1_motion_thread.join()
            self.robot2_impedance_motion.finish()
            self.robot2_motion_thread.join()
        elif self.control_type == "joint":
            self.robot1_WaypointMotionQPose.finish()
            self.robot1_motion_thread.join()
            self.robot2_WaypointMotionQPose.finish()
            self.robot2_motion_thread.join()


        time.sleep(0.5)
        # Move to home
        threading1 = self.robot1.move_async(JointMotion(HOME_JOINTS_l))
        threading2 = self.robot2.move_async(JointMotion(HOME_JOINTS_r))
        # threading3 = Thread(target=self.gripper1.open) #move(0.07)
        threading3 = Thread(target=self.gripper1.move, args=(0.07,))
        threading3.start()
        # threading4 = Thread(target=self.gripper2.open)
        threading4 = Thread(target=self.gripper2.move, args=(0.07,))
        threading4.start()

        threading1.join()
        threading2.join()
        threading3.join()
        threading4.join()

        return None
    


class WaypointMotionQPose():
    def __init__(self, robot: Robot):
        self.robot = robot
        self._stop_event = threading.Event()
        self._next_waypoint = None
        self._waypoint_lock = threading.Lock()
        self._waypoint_condition = Condition(self._waypoint_lock)

    def move(self, motion):
        self.robot.move(motion)
        # print(f'moving to {motion}')

    def move_async(self, initial_motion: JointMotion) -> Thread:
        p = Thread(target=self._move_and_wait, args=(initial_motion,), daemon=True)
        p.start()
        time.sleep(0.001)  # Sleep one control cycle
        return p

    def _move_and_wait(self, initial_motion: JointMotion):
        current_motion = initial_motion
        while not self._stop_event.is_set():
            self.move(current_motion)
            with self._waypoint_condition:
                while self._next_waypoint is None and not self._stop_event.is_set():
                    self._waypoint_condition.wait()
                if self._next_waypoint:
                    current_motion = self._next_waypoint
                    self._next_waypoint = None
        # print("Thread finishing")

    def set_next_waypoint(self, next_motion: JointMotion):
        with self._waypoint_condition:
            self._next_waypoint = next_motion
            self._waypoint_condition.notify()

    def finish(self):
        with self._waypoint_condition:
            self._stop_event.set()
            self._waypoint_condition.notify_all()