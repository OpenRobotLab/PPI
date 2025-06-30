from scipy.spatial.transform import Rotation as R
from utils.env import PPIRealEnv
import numpy as np
import time
import os
from pdb import set_trace
import pickle as pkl
from utils.controllers.tele_controller import TeleController
import cv2
from PIL import Image
import threading
import torch
from utils.transform_utils import xyzypr2xyzquat

PI = np.pi
# TODO: modify your path to PPI
path_to_PPI = "/PATH/TO/PPI"

class Control_ws():
    def __init__(self, task_name, episode_id, control_freq, instruction, env, controller):
        self.task_name = task_name
        self.episode_id = episode_id
        self.control_freq = control_freq
        self.env = env
        self.controller = controller
        self.instruction = instruction
        self.done = False

    def control_loop(self):
        while True:
            ############################################ each timestep ############################################s
            torch.cuda.synchronize() 
            t1 = time.time()

            robot1_ee_pose, robot2_ee_pose, robot1_q_pose, robot2_q_pose, robot1_gripper_open, robot2_gripper_open = self.env.get_robot_state()

            image_size = (640, 480)
            # t1_image = time.time()

            rgb_image, depth_image, intr, extr, is_keyframe, self.done = self.env.get_camera_data(image_size ,True)

            obs = {"robot1_ee_pose": robot1_ee_pose, # obs['robot1_ee_pose']
                   "robot2_ee_pose": robot2_ee_pose, # obs['robot2_ee_pose']
                   "robot1_q_pose": robot1_q_pose, # obs['robot1_q_pose']
                   "robot2_q_pose": robot2_q_pose, # obs['robot2_q_pose']
                   "robot1_gripper_open": robot1_gripper_open, # obs['robot1_gripper_open']
                   "robot2_gripper_open": robot2_gripper_open, # obs['robot2_gripper_open']
                   
                   "rgb_image": rgb_image, # obs['rgb_image']['left']
                   "depth_image": depth_image, # obs['depth_image']['left']
                   "intr": intr, # obs['intr']['left']
                   "extr": extr, # obs['extr']['left']
                   "is_keyframe": is_keyframe, # true or false
                   "instruction": self.instruction}
            

            robot1_action, robot2_action, gripper1_open, gripper2_open = self.controller(obs) # TeleController
            # print(f"robot1_action: {robot1_action}")
            # print(f"robot2_action: {robot2_action}")
            # set_trace()

            obs["robot1_action_ee_pose"] = robot1_action
            obs["robot2_action_ee_pose"] = robot2_action
            obs["robot1_action_gripper1_open"] = gripper1_open
            obs["robot2_action_gripper2_open"] = gripper2_open
            
            if self.env.flag_start:
                self.episode_obs.append(obs)

            torch.cuda.synchronize()
            t2 = time.time()
            sleep_left = 1 / self.control_freq - (t2 - t1)
            if sleep_left > 0:
                # print(f'sleep_left: {sleep_left}')
                time.sleep(sleep_left)
            else:
                print(f'control loop is too slow: {1 / (t2 - t1)}')
            ############################################ each timestep ############################################s

            self.env.step(robot1_action, robot2_action, gripper1_open, gripper2_open)

            if self.done:
                self.env.shutdown()
                self.done = False
                break

    def process(self):
        self.env.start()
        time.sleep(1)

        self.episode_obs = []
        
        self.control_loop()

        if self.env.flag_start:
            while(1):
                usr_input = input("Press Enter to \033[94msave data\033[0m, input 'q' to discard: ")
                if usr_input == '':
                    self.save_data()
                    print(f"Save data to {self.task_name}/episode{self.episode_id:04d}")
                    self.episode_id += 1
                    break
                elif usr_input == 'q':
                    print(f"Discard data of {self.task_name}/episode{self.episode_id:04d}")
                    break
                else:
                    print("Invalid input. Please input again.")
                    continue

        
        
    def save_data(self):
        '''
        obs = {"robot1_ee_pose": robot1_ee_pose, # obs['robot1_ee_pose']
               "robot2_ee_pose": robot2_ee_pose, # obs['robot2_ee_pose']
               "robot1_q_pose": robot1_q_pose, # obs['robot1_q_pose']
               "robot2_q_pose": robot2_q_pose, # obs['robot2_q_pose']
               "robot1_gripper_open": robot1_gripper_open, # obs['robot1_gripper_open']
               "robot2_gripper_open": robot2_gripper_open, # obs['robot2_gripper_open']
               "rgb_image": rgb_image, # obs['rgb_image']['left']
               "depth_image": depth_image, # obs['depth_image']['left']
               "intr": intr, # obs['intr']['left']
               "extr": extr, # obs['extr']['left']
               "is_keyframe": is_keyframe, # true or false
               "instruction": self.instruction}
        '''

        episode_folder = f"{path_to_PPI}/real_data/training_raw/{self.task_name}/episode{self.episode_id:04d}"

        if not os.path.exists(episode_folder):
            os.makedirs(episode_folder)

        threads_num = 5
        steps = len(self.episode_obs)
        threads = []
        for i in range(threads_num):
            start = i * steps // threads_num
            end = (i + 1) * steps // threads_num
            if i == threads_num - 1:
                end = steps
            thread = threading.Thread(target=self.save_multi_timesteps, args=(start, end, episode_folder))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

    def save_multi_timesteps(self, start, end, episode_folder):
        for timestep in range(start, end):
            # discard the first timestep, which is unstable
            if timestep == 0:
                continue
            self.save_one_timestep(timestep, episode_folder)
        
    def save_one_timestep(self, timestep, episode_folder):
        rgb_image = self.episode_obs[timestep]["rgb_image"]
        depth_image = self.episode_obs[timestep]["depth_image"]

        intr = self.episode_obs[timestep]["intr"]
        extr = self.episode_obs[timestep]["extr"]
        is_keyframe = self.episode_obs[timestep]["is_keyframe"]
        
        robot_state = {
            "robot1_ee_pose": self.episode_obs[timestep]["robot1_ee_pose"], # quaternion
            "robot2_ee_pose": self.episode_obs[timestep]["robot2_ee_pose"],
            "robot1_q_pose": self.episode_obs[timestep]["robot1_q_pose"],
            "robot2_q_pose": self.episode_obs[timestep]["robot2_q_pose"],
            "robot1_gripper_open": self.episode_obs[timestep]["robot1_gripper_open"],
            "robot2_gripper_open": self.episode_obs[timestep]["robot2_gripper_open"],
        }


        robot_action = {
            "robot1_action_ee_pose": xyzypr2xyzquat(self.episode_obs[timestep]["robot1_action_ee_pose"]),
            "robot2_action_ee_pose": xyzypr2xyzquat(self.episode_obs[timestep]["robot2_action_ee_pose"]),
            "robot1_action_gripper1_open": self.episode_obs[timestep]["robot1_action_gripper1_open"],
            "robot2_action_gripper2_open": self.episode_obs[timestep]["robot2_action_gripper2_open"],
        }
        

        instruction = self.episode_obs[timestep]["instruction"]

        other_data = {"robot_state": robot_state, 
                        "robot_action": robot_action, 
                        "intr": intr,
                        "extr": extr,
                        "is_keyframe": is_keyframe,
                        "instruction": instruction}

        timestep_folder = f"{episode_folder}/steps/{timestep-1:04d}"
        if not os.path.exists(timestep_folder):
            os.makedirs(timestep_folder)

        for camera in ['left', 'right', 'head']:
            rgb_filename = f'{timestep_folder}/{camera}_rgb.jpg'
            depth_filename = f'{timestep_folder}/{camera}_depth_x10000_uint16.png'


            cv2.imwrite(rgb_filename, np.asanyarray(cv2.cvtColor(rgb_image[camera], cv2.COLOR_RGB2BGR)))
            cv2.imwrite(depth_filename, depth_image[camera])

        other_data_filename = f"{timestep_folder}/other_data.pkl"
        with open(other_data_filename, 'wb') as f:
            pkl.dump(other_data, f)

def main():

    control_freq = 15
    # TODO: modify the instruction
    instruction = "handover and insert the plate"
    task_name = '_'.join(instruction.split())

    env = PPIRealEnv(robot_ip1="172.16.0.2",
            robot_ip2="172.16.1.2",
            control_freq=control_freq,
            control_type="pose",
            left_extr_path=f'{path_to_PPI}/real_world_deployment/calibration_results/view1_left_calibration.json',
            right_extr_path=f'{path_to_PPI}/real_world_deployment/calibration_results/view1_right_calibration.json',
            head_extr_path=f'{path_to_PPI}/real_world_deployment/calibration_results/view3_head_2_leftbase_calibration.json',
            if_tele=True,
            trans_stiffness=300,
            rot_stiffness=30,
            )
    
    controller = TeleController(control_type="pose",
                                env=env)
    # TODO: modify the start_episode_id of data-collection
    start_episode_id = 0
    control_ws = Control_ws(task_name, start_episode_id, control_freq, instruction, env, controller)
    while(1):
        print(f"=================== Episode {control_ws.episode_id} ===================")
        control_ws.process()
        
        user_input = input("Press Enter to continue, input 'q' to quit: ")
        if user_input == 'q':
            break
        control_ws.controller.update_init_action_pose()
    
    # stop the cameras
    for pipeline in env.multi_camera.pipelines:
        pipeline.stop()

    
        
if __name__ == "__main__":
    main()

