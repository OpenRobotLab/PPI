import os
import pickle
from PIL import Image
import numpy as np
from pdb import set_trace

class GetDataKeyframe():
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

    def process_episodes(self, start, end, skip_ep, kp_num):
        root = None

        state = []
        action = []
        language = []
        point_cloud = None
        pcd_paths = []
        dino_paths = []
        episode_ends = []
        keyframe_indices = []
        object_pose = []
        for episode in range(start, end + 1):
            episode_folder = os.path.join(self.data_path, f'episode{episode}')
            low_dim_obs_path = os.path.join(episode_folder, 'low_dim_obs.pkl')
            lang_path = os.path.join(episode_folder, 'variation_descriptions.pkl')

            if os.path.exists(lang_path):
                lang = self.read_pkl(lang_path)[0]
            if os.path.exists(self.lang_emb_path):
                instruction_embedding_dict = self.read_pkl(self.lang_emb_path)
                this_lang = instruction_embedding_dict[lang]
            if episode in skip_ep:
                continue

            if os.path.exists(low_dim_obs_path):
                low_dim_obs = self.read_pkl(low_dim_obs_path)

                stopping_delta = 0.1
                episode_keypoints = self.keypoint_discovery_bimanual(low_dim_obs, episode, stopping_delta,kp_num)

                if len(episode_ends) == 0:
                    for keyframe in episode_keypoints:
                        keyframe_indices.append(keyframe)
                    episode_ends.append(len(low_dim_obs))
                else:
                    for keyframe in episode_keypoints:
                        keyframe_indices.append(keyframe + episode_ends[-1])
                    episode_ends.append(len(low_dim_obs) + episode_ends[-1])

                for i in range(len(low_dim_obs)):           
                    gripper_pose = np.concatenate([
                                                low_dim_obs[i].left.gripper_pose,
                                                low_dim_obs[i].right.gripper_pose
                                                ])
                    gripper_state = np.array([
                                                low_dim_obs[i].left.gripper_open, 
                                                low_dim_obs[i].right.gripper_open
                                                ])

                    if i >= episode_keypoints[0] and len(episode_keypoints)>1:
                        episode_keypoints.pop(0)
                    #print(episode_keypoints)
                    keyframe_id = episode_keypoints[0]

                    keyframe_pose = np.concatenate([
                                                low_dim_obs[keyframe_id].left.gripper_pose,
                                                low_dim_obs[keyframe_id].right.gripper_pose
                                                ])
                    keyframe_state = np.array([
                                                low_dim_obs[keyframe_id].left.gripper_open, 
                                                low_dim_obs[keyframe_id].right.gripper_open
                                                ])
                    current_object_pose = np.concatenate([
                                                low_dim_obs[keyframe_id].object_6d_pose['position'],
                                                low_dim_obs[keyframe_id].object_6d_pose['quaternion']
                                                ])

                    current_state = np.concatenate([gripper_pose, gripper_state]) # (7 + 1) * 2 = 16
                    current_action = np.concatenate([keyframe_pose, keyframe_state]) # (7 + 1) * 2 = 16
                    state.append(current_state)
                    action.append(current_action)
                    language.append(this_lang)
                    pcd_paths.append(np.array([episode,i]))
                    dino_paths.append(np.array([episode,i]))
                    object_pose.append(current_object_pose)

            else:
                print(f"Warning: {low_dim_obs_path} does not exist")

        meta = dict()
        data = dict()
        meta["episode_ends"] = np.array(episode_ends)
        meta["keyframe_indices"] = np.array(keyframe_indices)
        data["state"] = np.array(state)
        data["action"] = np.array(action)
        data["point_cloud"] = np.array(pcd_paths)
        data["dino_feature"] = np.array(dino_paths)
        data["lang"] = np.array(language)
        data["object_pose"] = np.array(object_pose)

        root = {
                'meta': meta,
                'data': data
                }

        return root

    def keypoint_discovery_bimanual(self, low_dim_obs, episode, stopping_delta=0.1,total_kp=10):
        episode_keypoints = []
        right_prev_gripper_open = low_dim_obs[0].right.gripper_open
        left_prev_gripper_open = low_dim_obs[0].left.gripper_open
        stopped_buffer = 0
        for i, obs in enumerate(low_dim_obs):
            right_stopped = self._is_stopped_right(low_dim_obs, i, obs.right, stopping_delta)
            left_stopped = self._is_stopped_left(low_dim_obs, i, obs.left, stopping_delta)
            stopped = (stopped_buffer <= 0) and right_stopped and left_stopped
            stopped_buffer = 10 if stopped else stopped_buffer - 1
            # if change in gripper, or end of episode.
            last = i == (len(low_dim_obs) - 1)
            right_state_changed = obs.right.gripper_open != right_prev_gripper_open
            left_state_changed = obs.left.gripper_open != left_prev_gripper_open
            state_changed = right_state_changed or left_state_changed
            if i != 0 and (state_changed or last or stopped):
                episode_keypoints.append(i)

            right_prev_gripper_open = obs.right.gripper_open
            left_prev_gripper_open = obs.left.gripper_open
        if (
            len(episode_keypoints) > 1
            and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
        ):
            episode_keypoints.pop(-2)
        # print(f"Found {len(episode_keypoints)} keypoints in episode{episode}.")
        
        remaining_indices = np.array([i for i in range(len(low_dim_obs)) if i not in episode_keypoints])
        indices_to_sample = np.linspace(0, len(remaining_indices) - 1, num=total_kp-len(episode_keypoints), dtype=int)
        extra_episode_keypoints = remaining_indices[indices_to_sample]
        
        episode_keypoints.extend(extra_episode_keypoints)
        episode_keypoints.sort()
        # print(f"Totally {len(episode_keypoints)} keypoints in episode{episode}.")

        return episode_keypoints

    def _is_stopped_right(self, demo, i, obs, delta=0.1):
        next_is_not_final = i == (len(demo) - 2)
        gripper_state_no_change = i < (len(demo) - 2) and (
            obs.gripper_open == demo[i + 1].right.gripper_open
            and obs.gripper_open == demo[i - 1].right.gripper_open
            and demo[i - 2].right.gripper_open == demo[i - 1].right.gripper_open
        )
        small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
        return small_delta and (not next_is_not_final) and gripper_state_no_change


    def _is_stopped_left(self, demo, i, obs, delta=0.1):
        next_is_not_final = i == (len(demo) - 2)
        gripper_state_no_change = i < (len(demo) - 2) and (
            obs.gripper_open == demo[i + 1].left.gripper_open
            and obs.gripper_open == demo[i - 1].left.gripper_open
            and demo[i - 2].left.gripper_open == demo[i - 1].left.gripper_open
        )
        small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
        return small_delta and (not next_is_not_final) and gripper_state_no_change
