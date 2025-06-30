import torch
from ppi.model.common.normalizer import LinearNormalizer
from ppi.dataset.base_dataset import BaseDataset
import numpy as np
from pdb import set_trace
import pickle
from typing import Dict
import os
from ppi.common.get_data_keyframe_continuous_real import GetDataKeyframeContinuousReal

def main():
    prediction_type="keyframe_continuous"    #or keyframe
    # TODO: change the task name
    task_name = "handover_and_insert_the_plate"
    # task_name = "wipe_the_plate" 
    # task_name = "press_the_bottle"
    # task_name = "scan_the_bottle"
    # task_name = 'wear_the_scarf' 
    
    # TODO: change the path, 'training_processed' for normal setting, 'training_processed_simple' for simple setting
    pcd_path = f'real_data/training_processed/point_cloud/{task_name}'
    dino_path = f'real_data/training_processed/dino_feature/{task_name}'
    pointflow_path = f'real_data/training_processed/point_flow/{task_name}'
    skip_ep=[]

    if prediction_type=="keyframe_continuous":
        gd = GetDataKeyframeContinuousReal(
                data_path=f'real_data/training_raw/{task_name}', 
                lang_emb_path='pretrained_models/instruction_embeddings_real.pkl'
                )
    # TODO: change the episode number
    demo_num = 20
    root = gd.process_episodes(0, demo_num-1, skip_ep) 
    data = root['data']
    point_clouds = []
    # TODO: rgb_pcd_rps3072 for normal setting, rgb_pcd_rps512 for simple setting
    pcd_type = 'rgb_pcd_rps3072'
    dino_num = int(pcd_type.split('ps')[-1])
    # TODO: 200 for normal setting, 50 for simple setting
    point_flow_type = 'world_ordered_rps200'
    point_num=200

    dino_features = []
    dino_feature_tensor = torch.zeros(int(data['point_cloud'].shape[0]/3),dino_num,384,device='cuda:2')
    j=0
    print("start dino feature")
    for episode, step in data['dino_feature']:
        dino_feature_path = os.path.join(dino_path, f'episode{episode}/{pcd_type}/step{step:03d}.npy')
        dino_feature_data = np.load(dino_feature_path)
        dino_feature_tensor[j] = torch.tensor(dino_feature_data,device='cuda:1')
        j=j+1
        if j==int(data['point_cloud'].shape[0]/3)-1:
            break

    j=0
    print("start point flow")
    point_flow_tensor = torch.zeros(int(data['point_flow'].shape[0]),point_num,3,device='cuda:0')
    for episode, step in data['point_flow']:
        point_flow_path = os.path.join(pointflow_path, f'episode{episode}/{point_flow_type}/step{step:03d}.npy')
        point_flow_data = np.load(point_flow_path)
        point_flow_tensor[j] = torch.tensor(point_flow_data,device='cuda:0')
        j=j+1

    j=0    
    print("start initial point flow")
    initial_point_flow_tensor = torch.zeros(int(data['initial_point_flow'].shape[0]),point_num,3,device='cuda:0')  
    for episode, step in data['initial_point_flow']:
        initial_point_flow_path = os.path.join(pointflow_path, f'episode{episode}/{point_flow_type}/step{step:03d}.npy')
        initial_point_flow_data = np.load(initial_point_flow_path)
        initial_point_flow_tensor[j] = torch.tensor(initial_point_flow_data,device='cuda:0')
        j=j+1
    
    point_cloud_tensor = torch.zeros(data['point_cloud'].shape[0],dino_num,6,device='cuda:0')
    i = 0
    print("start point cloud")
    for episode, step in data['point_cloud']:
        pointcloud_path = os.path.join(pcd_path, f'episode{episode}/{pcd_type}/step{step:03d}.npy')
        point_cloud_data = np.load(pointcloud_path)
        point_cloud_tensor[i] = torch.tensor(point_cloud_data,device='cuda:0')
        i=i+1
    
    mode='limits'
    data4norm = {
        'action': torch.tensor(data["action"],device='cuda:0'),
        'agent_pos':  torch.tensor(data["state"],device='cuda:0'),
        'point_cloud': point_cloud_tensor,
        'dino_feature': dino_feature_tensor,
        'lang': torch.tensor(data["lang"],device='cuda:0'),
        'point_flow': point_flow_tensor,
        'initial_point_flow': initial_point_flow_tensor
    }
    print("start fit()")
    #set_trace()
    normalizer = LinearNormalizer()
    normalizer.fit(data=data4norm, last_n_dims=1, mode=mode)
    # TODO: change the path, 'training_processed' for normal setting, 'training_processed_simple' for simple setting
    stats_filepath = f'real_data/training_processed/norm_stats/norm_stats_{task_name}_{pcd_type}_{prediction_type}_{point_flow_type}.pth'
    print("start saving")
    torch.save(normalizer.state_dict(), stats_filepath)
    print(f"saved to {stats_filepath}")
    
if __name__ == "__main__":
    main()