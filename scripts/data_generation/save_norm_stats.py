import torch
from ppi.model.common.normalizer import LinearNormalizer
from ppi.dataset.base_dataset import BaseDataset
import numpy as np
from pdb import set_trace
import pickle
from typing import Dict
import os
from ppi.common.get_data_continuous import GetDataContinuous
from ppi.common.get_data_keyframe import GetDataKeyframe
from ppi.common.get_data_keyframe_continuous import GetDataKeyframeContinuous

# 示例使用
def main():
    # no need to change for the ablation tests of 'keyframe' and 'continuous'
    prediction_type="keyframe_continuous"    #or keyframe

    task_name = 'bimanual_lift_ball'
    # task_name = 'bimanual_push_box'
    # task_name = 'bimanual_put_item_in_drawer'
    # task_name = 'bimanual_lift_tray'
    # task_name = 'bimanual_pick_laptop'
    # task_name = 'bimanual_handover_item_easy'
    # task_name = "bimanual_sweep_to_dustpan"

    # TODO
    pcd_path=f'data/training_processed/point_cloud/{task_name}/all_variations/episodes'
    dino_path=f'data/training_processed/dino_feature/{task_name}/all_variations/episodes'
    pointflow_path = f'data/training_processed/point_flow/{task_name}/all_variations/episodes'
    skip_ep=[]
    if prediction_type=="continuous":
        gd = GetDataContinuous(
                data_path=f'data/training_raw/{task_name}/all_variations/episodes', 
                lang_emb_path='data/training_processed/instruction_embeddings.pkl'
                )
    elif prediction_type=="keyframe":
        gd = GetDataKeyframe(
                data_path=f'data/training_raw/{task_name}/all_variations/episodes', 
                lang_emb_path='data/training_processed/instruction_embeddings.pkl'
                )
    elif prediction_type=="keyframe_continuous":
        gd = GetDataKeyframeContinuous(
                data_path=f'data/training_raw/{task_name}/all_variations/episodes', 
                lang_emb_path='data/training_processed/instruction_embeddings.pkl'
                )
        root = gd.process_episodes(0, 99, skip_ep, 10)
    data = root['data']
    point_clouds = []
    pcd_type = 'rgb_pcd_rps6144'
    point_flow_type = 'world_ordered_rps200'

    # set_trace()
    point_num=200
    
    j=0
    print("start point flow")
    point_flow_tensor = torch.zeros(int(data['point_flow'].shape[0]),point_num,3,device='cuda:5')
    for episode, step in data['point_flow']:
        point_flow_path = os.path.join(pointflow_path, f'episode{episode}/{point_flow_type}/step{step:03d}.npy')
        point_flow_data = np.load(point_flow_path)
        # dino_features.append(dino_feature_data)
        point_flow_tensor[j] = torch.tensor(point_flow_data,device='cuda:5')
        j=j+1
        print(point_flow_path)
    j=0    
    print("start initial point flow")
    initial_point_flow_tensor = torch.zeros(int(data['initial_point_flow'].shape[0]),point_num,3,device='cuda:5')  
    for episode, step in data['initial_point_flow']:
        initial_point_flow_path = os.path.join(pointflow_path, f'episode{episode}/{point_flow_type}/step{step:03d}.npy')
        initial_point_flow_data = np.load(initial_point_flow_path)
        # dino_features.append(dino_feature_data)
        initial_point_flow_tensor[j] = torch.tensor(initial_point_flow_data,device='cuda:5')
        j=j+1
    
    point_cloud_tensor = torch.zeros(data['point_cloud'].shape[0],6144,6,device='cuda:6')
    i = 0
    for episode, step in data['point_cloud']:
        pointcloud_path = os.path.join(pcd_path, f'episode{episode}/{pcd_type}/step{step:03d}.npy')
        point_cloud_data = np.load(pointcloud_path)
        point_cloud_tensor[i] = torch.tensor(point_cloud_data,device='cuda:6')
        i=i+1
    print("finish point cloud")
    dino_features = []

    dino_feature_tensor = torch.zeros(int(data['point_cloud'].shape[0]/4),6144,384,device='cuda:7')
    j=0
    for episode, step in data['dino_feature']:
        dino_feature_path = os.path.join(dino_path, f'episode{episode}/{pcd_type}/step{step:03d}.npy')
        dino_feature_data = np.load(dino_feature_path)
        dino_feature_tensor[j] = torch.tensor(dino_feature_data,device='cuda:7')
        j=j+1
        if j==int(data['point_cloud'].shape[0]/4)-1:
            break
    mode='limits'
    data4norm = {
        'action': torch.tensor(data["action"],device='cuda:5'),
        'agent_pos':  torch.tensor(data["state"],device='cuda:5'),
        'point_cloud': point_cloud_tensor,
        'dino_feature': dino_feature_tensor,
        'lang': torch.tensor(data["lang"],device='cuda:5'),
        'point_flow': point_flow_tensor,
        'initial_point_flow': initial_point_flow_tensor
    }
    print("start fit()")
    #set_trace()
    normalizer = LinearNormalizer()
    normalizer.fit(data=data4norm, last_n_dims=1, mode=mode)
    # TODO
    stats_dir = "data/training_processed/norm_stats"
    os.makedirs(stats_dir, exist_ok=True)
    stats_filepath = f"{stats_dir}/norm_stats_{task_name}_{pcd_type}_{prediction_type}_{point_flow_type}_new.pth"

    print("start saving")
    torch.save(normalizer.state_dict(), stats_filepath)
    print(f"saved to {stats_filepath}")

    
if __name__ == "__main__":
    main()