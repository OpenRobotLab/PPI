import os
import pickle
from PIL import Image
import numpy as np
import zarr
import glob
from pdb import set_trace

class GetDataKeyframeContinuousReal():
    def __init__(self, data_path=None, lang_emb_path=None):
        self.data_path = data_path
        self.lang_emb_path = lang_emb_path
        
    def read_pkl(self, file_path):
        with open(file_path, 'rb') as f:
            obs = pickle.load(f)
        return obs

    def read_pngs(self, folder_path):
        png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        images = {}
        for png_file in png_files:
            img_path = os.path.join(folder_path, png_file)
            with Image.open(img_path) as img:
                images[png_file] = np.array(img)
        return images

    def process_episodes(self, start, end, skip_ep, hist_num=1):
        root = None

        state = []
        action = []
        language = []
        point_cloud = None
        pcd_paths = []
        dino_paths = []
        episode_ends = []
        keyframe_indices = []
        initial_point_flow=[]
        openess_indices = []

        gap = 9 # TODO: balance the control frequency

        for episode in range(start, end + 1):
            episode_folder = f"{self.data_path}/episode{episode:04d}"
            other_data_0_path = f"{episode_folder}/steps/0000/other_data.pkl"
            other_data_0 = self.read_pkl(other_data_0_path)
            lang = other_data_0['instruction']
            robot1_gripper_open_last = other_data_0['robot_state']['robot1_gripper_open']
            robot2_gripper_open_last = other_data_0['robot_state']['robot2_gripper_open']

            if os.path.exists(self.lang_emb_path):
                instruction_embedding_dict = self.read_pkl(self.lang_emb_path)
                this_lang = instruction_embedding_dict[lang]

            if episode in skip_ep:
                print(f'skip episode: {episode}')
                continue
            print('Loading episode: ',episode)

            steps_dir = sorted(glob.glob(f"{episode_folder}/steps/*"))
            steps_num = len(steps_dir)

            # load keyframe
            keyframes = []
            for step in steps_dir:
                with open(f'{step}/other_data.pkl', 'rb') as f:
                    other_data = pickle.load(f)

                if other_data['is_keyframe']:
                    step_id = int(step.split('/')[-1])
                    keyframes.append(step_id)

            extracted_steps_num = 0
            extracted_steps_id = []
            for i in range(steps_num):         
                if i >= keyframes[-2]:
                    break
                if i % gap != 0 and i not in keyframes and (i + gap) not in keyframes:
                    continue

                extracted_steps_num += 1
                extracted_steps_id.append(i)

                other_data_path = f"{episode_folder}/steps/{i:04d}/other_data.pkl"
                other_data = self.read_pkl(other_data_path)

                action_data_path = f"{episode_folder}/steps/{min(i + gap, steps_num - 1):04d}/other_data.pkl"
                action_data = self.read_pkl(action_data_path)

                is_keyframe = other_data["is_keyframe"]
                if is_keyframe:
                    if len(episode_ends) == 0:
                        keyframe_indices.append(extracted_steps_num - 1)
                    else:
                        keyframe_indices.append(extracted_steps_num - 1 + episode_ends[-1])


                robot1_gripper_open = other_data['robot_state']['robot1_gripper_open']
                robot2_gripper_open = other_data['robot_state']['robot2_gripper_open']

                if robot1_gripper_open != robot1_gripper_open_last or robot2_gripper_open != robot2_gripper_open_last:
                    print(f"add openess_indices: {extracted_steps_num - 1}")
                    if len(episode_ends) == 0:
                        openess_indices.append(extracted_steps_num - 1)
                    else:
                        openess_indices.append(extracted_steps_num - 1 + episode_ends[-1])
                    robot1_gripper_open_last = robot1_gripper_open
                    robot2_gripper_open_last = robot2_gripper_open


                gripper_pose_state = np.array(other_data['robot_state']['robot1_ee_pose'] + other_data['robot_state']['robot2_ee_pose'])
                gripper_open_state = np.array([other_data['robot_state']['robot1_gripper_open'], other_data['robot_state']['robot2_gripper_open']])

                gripper_pose_action = np.array(action_data['robot_action']['robot1_action_ee_pose'] + action_data['robot_action']['robot2_action_ee_pose'])
                gripper_open_action = np.array([action_data['robot_action']['robot1_action_gripper1_open'], action_data['robot_action']['robot2_action_gripper2_open']])

                current_action = np.concatenate([gripper_pose_action, gripper_open_action]) # (7 + 1) * 2 = 16
                
                hist_state = None
                i_id_in_extracted_steps_id = extracted_steps_id.index(i)
                for k in range(hist_num):
                    hist_id = max(0, i_id_in_extracted_steps_id - k)
                    other_data_path = f"{episode_folder}/steps/{extracted_steps_id[hist_id]:04d}/other_data.pkl"
                    other_data = self.read_pkl(other_data_path)
                    gripper_pose_state = np.array(other_data['robot_state']['robot1_ee_pose'] + other_data['robot_state']['robot2_ee_pose'])
                    gripper_open_state = np.array([other_data['robot_state']['robot1_gripper_open'], other_data['robot_state']['robot2_gripper_open']])
                    current_state = np.concatenate([gripper_pose_state, gripper_open_state]) # (7 + 1) * 2 = 16
                    if hist_state is None:
                        hist_state = current_state
                    else:
                        hist_state = np.concatenate([hist_state, current_state])

                state.append(hist_state)

                action.append(current_action)
                language.append(this_lang)
                pcd_paths.append(np.array([episode,i]))
                dino_paths.append(np.array([episode,i]))
                initial_point_flow.append(np.array([episode,0]))

            if len(episode_ends) == 0:
                episode_ends.append(extracted_steps_num)
            else:
                episode_ends.append(extracted_steps_num + episode_ends[-1])
            

        meta = dict()
        data = dict()

        print('episode_ends: ',episode_ends)
        meta["episode_ends"] = np.array(episode_ends)
        meta["keyframe_indices"] = np.array(keyframe_indices)


        meta["openess_indices"] = np.array(openess_indices)
        data["state"] = np.array(state)
        data["action"] = np.array(action)
        data["point_cloud"] = np.array(pcd_paths)
        data["dino_feature"] = np.array(dino_paths)
        data["point_flow"] = np.array(pcd_paths)
        data["lang"] = np.array(language)
        data["initial_point_flow"] = np.array(initial_point_flow)

        root = {
                'meta': meta,
                'data': data
                }

        return root
