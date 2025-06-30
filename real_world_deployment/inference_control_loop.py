from scipy.spatial.transform import Rotation as R
from utils.env import PPIRealEnv
import numpy as np
import time
import os
from pdb import set_trace
import pickle as pkl
from utils.controllers.inference_controller import InferenceController
import cv2
import threading
import hydra
import pathlib
import torch
from tqdm import tqdm

PI = np.pi
# TODO: modify your path to PPI
path_to_PPI = "/PATH/TO/PPI"

class Control_ws():
    def __init__(self, task_name, episode_id, control_freq, instruction, env, controller, cfg):
        self.task_name = task_name
        self.episode_id = episode_id
        self.control_freq = control_freq
        self.env = env
        self.controller = controller
        self.instruction = instruction
        self.done = False
        self.cfg = cfg

    def control_loop(self):
        # TODO: modify the hist_num of proprioception
        # hist_num = self.cfg.task.dataset.hist_num
        hist_num = 3
        self.is_first_frame = True
        skip_first = True
        # while True:
        for _ in tqdm(range(105)):
            # print("==============================================")

            ############################################ each timestep ############################################s
            torch.cuda.synchronize() 
            t1 = time.time()

            robot1_ee_pose, robot2_ee_pose, robot1_q_pose, robot2_q_pose, robot1_gripper_open, robot2_gripper_open = self.env.get_robot_state()
            torch.cuda.synchronize() 
            t2 = time.time()
            print(f'get robot state time: {t2 - t1}')

            image_size = (640, 480)
            rgb_image, depth_image, intr, extr, is_keyframe, self.done = self.env.get_camera_data(image_size, x10000_depth=False)

            torch.cuda.synchronize() 
            t3 = time.time()
            print(f'get camera data time: {t3 - t2}')

            current_obs = {"robot1_ee_pose": robot1_ee_pose, # obs['robot1_ee_pose']
                            "robot2_ee_pose": robot2_ee_pose, # obs['robot2_ee_pose']
                            "robot1_q_pose": robot1_q_pose, # obs['robot1_q_pose']
                            "robot2_q_pose": robot2_q_pose, # obs['robot2_q_pose']
                            "robot1_gripper_open": robot1_gripper_open, # obs['robot1_gripper_open']
                            "robot2_gripper_open": robot2_gripper_open, # obs['robot2_gripper_open']
                            
                            "head_rgb_image": rgb_image['head'], # obs['rgb_image']['left']
                            "head_depth_image": depth_image['head'], # obs['depth_image']['left']
                            "head_intr": intr['head'], # obs['intr']['left']
                            "head_extr": extr['head'], # obs['extr']['left']

                            "left_rgb_image": rgb_image['left'],
                            "left_depth_image": depth_image['left'],
                            "left_intr": intr['left'],
                            "left_extr": extr['left'],

                            "right_rgb_image": rgb_image['right'],
                            "right_depth_image": depth_image['right'],
                            "right_intr": intr['right'],
                            "right_extr": extr['right'],

                            "is_keyframe": is_keyframe, # true or false
                            "instruction": self.instruction
                            }
            torch.cuda.synchronize() 
            t4 = time.time()

            if skip_first:
                skip_first = False
                continue

            if self.is_first_frame:
                obs_hist = {}
                for key, value in current_obs.items():
                    if key == 'is_keyframe':
                        obs_hist[key] = np.array([value] * self.cfg.n_obs_steps, dtype=bool)
                    elif key == 'instruction':
                        obs_hist[key] = np.array([value] * self.cfg.n_obs_steps, dtype=str)
                    elif "robot" in key:
                        obs_hist[key] = np.stack([value] * hist_num)
                    else:
                        obs_hist[key] = np.stack([value] * self.cfg.n_obs_steps)
                self.is_first_frame = False
                # for key, value in obs_hist.items():
                #     print(f"key: {key}, value: {value.shape}")
                # set_trace()
                '''
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
                '''

            else:
                for key, value in obs_hist.items():
                    if key == 'is_keyframe':
                        obs_hist[key] = np.array(list(value[1:]) + [current_obs[key]], dtype=bool)
                    elif key == 'instruction':
                        obs_hist[key] = np.array(list(value[1:]) + [current_obs[key]], dtype=str)
                    else:
                        obs_hist[key] = np.concatenate((value[1:], np.expand_dims(np.array(current_obs[key]), axis=0)), axis=0)

            torch.cuda.synchronize() 
            t5 = time.time()
            # print(f'update obs_hist time: {t5 - t4}')

            robot1_action, robot2_action, gripper1_open, gripper2_open = self.controller(obs_hist,
                                                                                         self.cfg.n_obs_steps, 
                                                                                         hist_num) # TeleController
            
            torch.cuda.synchronize() 
            t6 = time.time()
            print(f'Process_obs_hist + predict_action time: {t6 - t5}')
            
            torch.cuda.synchronize()
            t2 = time.time()
            sleep_left = 1 / self.control_freq - (t2 - t1)
            if sleep_left > 0:
                time.sleep(sleep_left)
            ############################################ each timestep ############################################s

            self.env.step(robot1_action, robot2_action, gripper1_open, gripper2_open)

            torch.cuda.synchronize() 
            t7 = time.time()

            if self.done:
                break

        self.env.shutdown()

    def process(self):
        self.env.start()
        time.sleep(1)

        self.control_loop()

@hydra.main(
    version_base=None,
    # TODO: modify the config path
    config_path=f'{path_to_PPI}/real_world_deployment/ckpts/PPI/handover_and_insert_the_plate-ppi-032103_handover_and_insert_the_plate_simple_512_64_h3_s10_seed0',
    config_name='config'
)
def main(cfg):
    control_freq = 15/9 
    # TODO: modify the instruction
    instruction = "handover and insert the plate"

    task_name = '_'.join(instruction.split())
    # TODO: modify the ckpt path
    ckpt_path = f'{path_to_PPI}/real_world_deployment/ckpts/PPI/handover_and_insert_the_plate-ppi-032103_handover_and_insert_the_plate_simple_512_64_h3_s10_seed0/epoch4800_model.pth.tar'

    # TODO: modify the lang_emb_path path
    lang_emb_path = f'{path_to_PPI}/pretrained_models/instruction_embeddings_real.pkl'
    point_cloud_num = 3072
    # point_cloud_num = 512
    prompt_type = 'box'
    point_flow_sample_type = 'rps'
    sam_cameras = ['head']

    # TODO: modify the text_prompt
    ### single-obj:
    text_prompt = "A purple plate."
    pointflow_num = 200
    objs_and_pfl_num = None
    if_multi_objs = False

    ### multi-objs:
    # text_prompt = ["A yellow sponge.", "A purple plate."]
    # pointflow_num = 200
    # objs_and_pfl_num = {"sponge": 50, "plate": 150}
    # if_multi_objs = True
    
    env = PPIRealEnv(robot_ip1="172.16.0.2",
            robot_ip2="172.16.1.2",
            control_freq=control_freq,
            control_type="pose",
            left_extr_path=f'{path_to_PPI}/real_world_deployment/calibration_results/view1_left_calibration.json',
            right_extr_path=f'{path_to_PPI}/real_world_deployment/calibration_results/view1_right_calibration.json',
            head_extr_path=f'{path_to_PPI}/real_world_deployment/calibration_results/view3_head_2_leftbase_calibration.json',
            if_tele=True
            )
    
    t1 = time.time()
    cfg.policy.num_inference_steps = 20
    controller = InferenceController(ckpt_path = ckpt_path,
                                control_type="pose",
                                env=env,
                                cfg=cfg,
                                device=torch.device("cuda:0"),
                                lang_emb_path=lang_emb_path,
                                instruction=instruction,
                                point_cloud_num=point_cloud_num,
                                pointflow_num = pointflow_num,
                                text_prompt=text_prompt,
                                prompt_type=prompt_type,
                                point_flow_sample_type=point_flow_sample_type,
                                sam_cameras=sam_cameras,
                                if_multi_objs=if_multi_objs,
                                objs_and_pfl_num=objs_and_pfl_num,
                                )
    # set_trace()
    control_ws = Control_ws(task_name, 0, control_freq, instruction, env, controller, cfg)
    t2 = time.time()
    # print(f'init controller time: {t2 - t1}')

    while(1):
        set_trace()
        control_ws.process()
        set_trace()
        control_ws.is_first_frame = True
        control_ws.controller.reset()
        
    
    # stop the cameras
    for pipeline in env.multi_camera.pipelines:
        pipeline.stop()

if __name__ == "__main__":
    main()

