from typing import Dict
import torch
import numpy as np
import copy
from ppi.common.pytorch_util import dict_apply
from ppi.common.replay_buffer_real import ReplayBufferReal
from ppi.common.sampler_keyframe_continuous_real import (
    SequenceSamplerKeyframeContinuousReal, get_val_mask, downsample_mask)
from ppi.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from ppi.dataset.base_dataset import BaseDataset
from pdb import set_trace
class RealDataset(BaseDataset):
    def __init__(self,
            data_path, 
            pcd_path,
            lang_emb_path,
            dino_path,
            stats_filepath,
            point_flow_path,
            horizon_keyframe=1,
            horizon_continuous=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            start=0,
            end=3,
            pcd_fps=1024,
            skip_ep=None,
            kp_num=10,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            pcd_type = "2views_point_cloud_rps2048",
            prediction_type="continuous",
            point_flow_type="rps200",
            hist_num=1,
            add_openess_sampling=False,
            sample_num=6
            ):
        super().__init__()
        self.task_name = task_name
        self.stats_filepath = stats_filepath

        self.replay_buffer = ReplayBufferReal.getData_keyframe_continuous(
            data_path, pcd_path, lang_emb_path, dino_path,start=start, end=end, pcd_fps=pcd_fps, skip_ep=skip_ep, kp_num=kp_num, hist_num=hist_num, keys=['state', 'action', 'point_cloud', 'lang', 'dino_feature', 'point_flow','initial_point_flow'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)
        self.pcd_path = pcd_path
        self.dino_path = dino_path
        self.pcd_type = pcd_type
        self.point_flow_path = point_flow_path
        self.point_flow_type = point_flow_type

        self.sampler = SequenceSamplerKeyframeContinuousReal(
            replay_buffer=self.replay_buffer, 
            sequence_length_keyframe=horizon_keyframe,
            sequence_length_continuous=horizon_continuous,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            pcd_path = self.pcd_path,
            dino_path = self.dino_path,
            pcd_type = self.pcd_type,
            point_flow_path = self.point_flow_path,
            point_flow_type = self.point_flow_type,
            add_openess_sampling = add_openess_sampling,
            sample_num = sample_num
            )
        self.train_mask = train_mask
        self.horizon_keyframe = horizon_keyframe
        self.horizon_continuous = horizon_continuous
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.prediction_type = prediction_type

    def get_validation_dataset(self):
        val_set = copy.copy(self)

        val_set.sampler = SequenceSamplerKeyframeContinuousReal(
            replay_buffer=self.replay_buffer, 
            sequence_length_keyframe=self.horizon_keyframe,
            sequence_length_continuous=self.horizon_continuous,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            pcd_path = self.pcd_path,
            dino_path = self.dino_path,
            pcd_type = self.pcd_type,
            point_flow_path = self.point_flow_path,
            point_flow_type = self.point_flow_type,
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        state_dict = torch.load(self.stats_filepath, map_location=torch.device('cuda:0'))
        normalizer = LinearNormalizer()
        normalizer.load_state_dict(state_dict)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32)
        point_cloud = sample['point_cloud'][:,].astype(np.float32) 
        lang = sample['lang'][:,].astype(np.float32)
        dino_feature = sample['dino_feature'][:,].astype(np.float32)
        point_flow = sample['point_flow'][:,].astype(np.float32)
        initial_point_flow = sample['initial_point_flow'][:,].astype(np.float32)
        data = {
            'obs': {
                'point_cloud': point_cloud,
                'agent_pos': agent_pos,
                'lang': lang,
                'dino_feature': dino_feature,
                'initial_point_flow': initial_point_flow
            },
            'action': sample['action'].astype(np.float32), 
            'point_flow': point_flow
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

