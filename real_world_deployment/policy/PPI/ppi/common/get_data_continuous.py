import os
import pickle
from PIL import Image
import numpy as np
from pdb import set_trace

class GetDataContinuous():
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

    def process_episodes(self, start, end, skip_ep):
        root = None

        state = []
        action = []
        language = []
        point_cloud = None
        pcd_paths = []
        dino_paths = []
        episode_ends = []
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

                if len(episode_ends) == 0:
                    episode_ends.append(len(low_dim_obs))
                else:
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
                    current_state = np.concatenate([gripper_pose, gripper_state]) # (7 + 1) * 2 = 16
                    current_action = np.concatenate([gripper_pose, gripper_state]) # (7 + 1) * 2 = 16
                    state.append(current_state)
                    action.append(current_action)
                    language.append(this_lang)
                    pcd_paths.append(np.array([episode,i]))
                    dino_paths.append(np.array([episode,i]))
                
            else:
                print(f"Warning: {low_dim_obs_path} does not exist")
       
        meta = dict()
        data = dict()
        
        meta["episode_ends"] = np.array(episode_ends)
        data["state"] = np.array(state)
        data["action"] = np.array(action)
        data["point_cloud"] = np.array(pcd_paths)
        data["dino_feature"] = np.array(dino_paths)
        data["lang"] = np.array(language)

        root = {
                'meta': meta,
                'data': data
                }

        return root

if __name__ == "__main__":
    data_path = ''
    lang_emb_path = ''
    getdata = GetDataContinuous(data_path,lang_emb_path,)
    root = getdata.process_episodes(0, 3, skip_ep=[])
    print(root["data"]["state"].shape)
    print(root["data"]["action"].shape)
    print(root["data"]["point_cloud"].shape)
    print(root["data"]["dino_feature"].shape)
    print(root["data"]["lang"].shape)
    print(root["meta"]["episode_ends"])