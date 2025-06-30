from typing import Optional
import numpy as np
import numba
from ppi.common.replay_buffer import ReplayBuffer
from pdb import set_trace
import os


def create_indices(keyframe_indices:np.ndarray, 
    episode_ends:np.ndarray, 
    sequence_length_keyframe:int,
    sequence_length_continuous:int,
    episode_mask: np.ndarray,
    pad_before: int=0, pad_after: int=0,
    debug:bool=True,
    openess_indices:np.ndarray=None,
    add_openess_sampling:bool=False) -> np.ndarray: 
    sequence_length = sequence_length_keyframe + sequence_length_continuous
    episode_mask.shape == episode_ends.shape        
    pad_before = min(max(pad_before, 0), sequence_length-1)
    pad_after = min(max(pad_after, 0), sequence_length-1)
    
    indices = list()
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # skip episode
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        
        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            # set_trace()
            keyframe_indices_mask = (keyframe_indices > idx + start_idx) & (keyframe_indices < end_idx)
            
            keyframe_filtered_indices = keyframe_indices[keyframe_indices_mask]
 
            openess_indices_mask = (openess_indices > idx + start_idx) & (openess_indices < end_idx)
            
            openess_filtered_indices = openess_indices[openess_indices_mask]

            if keyframe_filtered_indices.size == 0:
                keyframe_filtered_indices = np.array([idx + start_idx])
                
            if openess_filtered_indices.size == 0:
                openess_filtered_indices = np.array([idx + start_idx])
            # print(f'idx: {idx} filtered_indices: {filtered_indices}')
            keyframe_padding = np.full(pad_after, keyframe_filtered_indices[-1])
            keyframe_padded_filtered_indices = np.concatenate((keyframe_filtered_indices, keyframe_padding))
                      
            continuous_indices = np.arange(idx + start_idx, idx + start_idx + sequence_length_continuous)
            continuous_indices_mask = (continuous_indices >= idx + start_idx) & (continuous_indices < end_idx)
            continuous_filtered_indices = continuous_indices[continuous_indices_mask]
            
            continuous_padding = np.full(pad_after, continuous_filtered_indices[-1])
            continuous_padded_filtered_indices = np.concatenate((continuous_filtered_indices, continuous_padding))

            precent_indices = np.concatenate((continuous_padded_filtered_indices[:sequence_length_continuous], keyframe_padded_filtered_indices[:sequence_length_keyframe]))
            # print(f"start_id:{idx + start_idx}")
            # print(f"precent_indices:{precent_indices}")
            # set_trace()
            distance_to_openess = openess_filtered_indices[0] - continuous_padded_filtered_indices[0]
            if add_openess_sampling:
                if distance_to_openess>0 and distance_to_openess<=10:
                    for i in range(2):
                        indices.append(precent_indices)
                
                if distance_to_openess>0 and distance_to_openess<=5:
                    for i in range(3):
                        indices.append(precent_indices)
            
            indices.append(precent_indices)
   
    indices = np.array(indices)
    # print('indices.shape: ', indices.shape)
    
    return indices

def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask, max_n, seed=0):
    # subsample training data
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask

class SequenceSamplerKeyframeContinuous:
    def __init__(self, 
        replay_buffer: ReplayBuffer, 
        sequence_length_keyframe:int,
        sequence_length_continuous:int,
        pad_before:int=0,
        pad_after:int=0,
        keys=None,
        key_first_k=dict(),
        episode_mask: Optional[np.ndarray]=None,
        pcd_path=None,
        dino_path=None,
        pcd_type = None,
        point_flow_path = None,
        point_flow_type = None,
        add_openess_sampling = False
        ):
        """
        key_first_k: dict str: int
            Only take first k data from these keys (to improve perf)
        """

        super().__init__()
        assert(sequence_length_keyframe + sequence_length_continuous >= 1)
        if keys is None:
            keys = list(replay_buffer.keys())

        keyframe_indices = replay_buffer.meta['keyframe_indices']
        openess_indices = replay_buffer.meta['openess_indices']
        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)
        indices = create_indices(keyframe_indices,
            episode_ends, 
            sequence_length_keyframe=sequence_length_keyframe,
            sequence_length_continuous=sequence_length_continuous, 
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=episode_mask,
            openess_indices = openess_indices,
            add_openess_sampling = add_openess_sampling
        )
        self.indices = indices 
        self.keys = list(keys) # prevent OmegaConf list performance problem
        self.sequence_length_keyframe = sequence_length_keyframe
        self.sequence_length_continuous = sequence_length_continuous
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k
        self.pcd_path = pcd_path
        self.dino_path = dino_path
        self.pcd_type = pcd_type
        self.point_flow_type = point_flow_type
        self.point_flow_path = point_flow_path
    
    def __len__(self):
        return len(self.indices)
        
    
    def sample_sequence(self, idx):
        indices = self.indices[idx]
        result = dict()
        for key in self.keys:
            input_arr = self.replay_buffer[key]

            if key == 'point_cloud':
                point_clouds = []

                pcd_path_idx = indices[0]
                episode, step = input_arr[pcd_path_idx]
                pointcloud_path = os.path.join(self.pcd_path, f'episode{episode}/{self.pcd_type}/step{step:03d}.npy')
                point_cloud_data = np.load(pointcloud_path)
                point_clouds.append(point_cloud_data)
                    
                data = np.array(point_clouds)
            elif key == 'dino_feature':
                dino_features = []
                dino_path_idx = indices[0]  
                episode, step = input_arr[dino_path_idx]
                dino_feature_path = os.path.join(self.dino_path, f'episode{episode}/{self.pcd_type}/step{step:03d}.npy')
                dino_feature_data = np.load(dino_feature_path)
                dino_features.append(dino_feature_data)  
                data = np.array(dino_features)
                # print(data.shape)   
            elif key== 'point_flow':
                point_flow = []
                indice_keyframe = indices[-self.sequence_length_keyframe:]
                for pointflow_path_idx in indice_keyframe:
                    episode, step = input_arr[pointflow_path_idx]
                    point_flow_path = os.path.join(self.point_flow_path, f'episode{episode}/{self.point_flow_type}/step{step:03d}.npy')
                    point_flow_data = np.load(point_flow_path)
                    point_flow.append(point_flow_data)    
                data = np.array(point_flow)  
            elif key == 'initial_point_flow':
                initial_point_flow = []
                pcd_path_idx = indices[0]
                episode, step = input_arr[pcd_path_idx]
                initial_point_flow_path = os.path.join(self.point_flow_path, f'episode{episode}/{self.point_flow_type}/step{step:03d}.npy')
                initial_point_flow_data = np.load(initial_point_flow_path)
                initial_point_flow.append(initial_point_flow_data)
                    
                data = np.array(initial_point_flow)
            else:
                data = input_arr[indices]

            result[key] = data
        return result