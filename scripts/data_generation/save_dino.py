import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
from PIL import Image
from PIL import Image
import pickle
from pdb import set_trace
import threading
import time
def project_points_coords(pts, Rt, K):
    """
    :param pts:  [pn,3]
    :param Rt:   [rfn,3,4]
    :param K:    [rfn,3,3]
    :return:
        coords:         [rfn,pn,2]
        invalid_mask:   [rfn,pn] 
        depth:          [rfn,pn,1]
    """
    pn = pts.shape[0]
    hpts = torch.cat([pts,torch.ones([pn,1],device=pts.device,dtype=pts.dtype)],1)  # [pn, 4]
    srn = Rt.shape[0]
    KRt = K @ Rt # rfn,3,4
    last_row = torch.zeros([srn,1,4],device=pts.device,dtype=pts.dtype)
    last_row[:,:,3] = 1.0
    H = torch.cat([KRt,last_row],1) # rfn,4,4
    pts_cam = H[:,None,:,:] @ hpts[None,:,:,None]
    pts_cam = pts_cam[:,:,:3,0]
    depth = pts_cam[:,:,2:]
    invalid_mask = torch.abs(depth)<1e-4
    depth[invalid_mask] = 1e-3
    pts_2d = pts_cam[:,:,:2]/depth
    return pts_2d, ~(invalid_mask[...,0]), depth

def interpolate_feats(feats, points, h=None, w=None, padding_mode='zeros', align_corners=False, inter_mode='bilinear'):
    """

    :param feats:   b,f,h,w
    :param points:  b,n,2
    :param h:       float
    :param w:       float
    :param padding_mode:
    :param align_corners:
    :param inter_mode:
    :return: feats_inter: b,n,f
    """
    b, _, ch, cw = feats.shape
    if h is None and w is None:
        h, w = ch, cw
    x_norm = points[:, :, 0] / (w - 1) * 2 - 1
    y_norm = points[:, :, 1] / (h - 1) * 2 - 1
    points_norm = torch.stack([x_norm, y_norm], -1).unsqueeze(1)
    feats_inter = F.grid_sample(feats, points_norm, mode=inter_mode, padding_mode=padding_mode, align_corners=align_corners).squeeze(2)      # srn,f,n
    feats_inter = feats_inter.permute(0,2,1)
    return feats_inter

class Fusion():
    def __init__(self, num_cam, feat_backbone='dinov2', device='cuda:0', dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        # hyper-parameters
        self.mu = 0.02
        # curr_obs_torch is a dict contains:
        # - 'dino_feats': (K, patch_h, patch_w, feat_dim) torch tensor, dino features
        # - 'depth': (K, H, W) torch tensor, depth images
        # - 'pose': (K, 4, 4) torch tensor, poses of the images
        # - 'K': (K, 3, 3) torch tensor, intrinsics of the images
        self.curr_obs_torch = {}
        self.H = -1
        self.W = -1
        self.num_cam = num_cam
        
        # dino feature extractor
        self.feat_backbone = feat_backbone
        if self.feat_backbone == 'dinov2':
            # TODO
            model_path = "pretrained_models/hub/checkpoints/dinov2_vits14_pretrain.pth"
            repo_path = "repos/dinov2"
            os.environ["TORCH_HOME"] = "pretrained_models"
            self.dinov2_feat_extractor = torch.hub.load(repo_path, 'dinov2_vits14',source='local',skip_validation=True)
            state_dict = torch.load(model_path,map_location=self.device)
            self.dinov2_feat_extractor.load_state_dict(state_dict)
            self.dinov2_feat_extractor.to(self.device)

        else:
            raise NotImplementedError
        self.dinov2_feat_extractor.eval()
    
        
    def eval(self, pts, return_names=['dino_feats'], return_inter=False):
        # :param pts: (N, 3) torch tensor in world frame
        # :param return_names: a set of {'dino_feats', 'mask'}
        # :return: output: dict contains:
        #          - 'dist': (N) torch tensor, dist to the closest point on the surface
        #          - 'dino_feats': (N, f) torch tensor, the features of the points
        #          - 'mask': (N, NQ) torch tensor, the query masks of the points
        #          - 'valid_mask': (N) torch tensor, whether the point is valid
        # curr_obs_torch is a dict contains:
        # - 'dino_feats': (K, patch_h, patch_w, feat_dim) torch tensor, dino features
        # - 'depth': (K, H, W) torch tensor, depth images
        # - 'pose': (K, 4, 4) torch tensor, poses of the images
        # - 'K': (K, 3, 3) torch tensor, intrinsics of the images
        try:
            assert len(self.curr_obs_torch) > 0
        except:
            print('Please call update() first!')
            exit()
        assert type(pts) == torch.Tensor
        assert len(pts.shape) == 2
        assert pts.shape[1] == 3
        
        # transform pts to camera pixel coords and get the depth
        pts_2d, valid_mask, pts_depth = project_points_coords(pts, self.curr_obs_torch['pose'], self.curr_obs_torch['K'])
        # torch.sum(torch.all(((pts_2d>=0)&(pts_2d<=256)),dim=-1),dim=1)
        pts_depth = pts_depth[...,0] # [rfn,pn]
        #set_trace()
        # get interpolated depth and features
        inter_depth = interpolate_feats(self.curr_obs_torch['depth'].unsqueeze(1),
                                        pts_2d,
                                        h = self.H,
                                        w = self.W,
                                        padding_mode='zeros',
                                        align_corners=True,
                                        inter_mode='nearest')[...,0] # [rfn,pn,1]

        dist = inter_depth - pts_depth # [rfn,pn]
        dist_valid = (inter_depth > 0.0) & valid_mask & (dist > -self.mu) # [rfn,pn]
        
        # distance-based weight
        dist_weight = torch.exp(torch.clamp(self.mu-torch.abs(dist), max=0) / self.mu) # [rfn,pn]
        
        dist = torch.clamp(dist, min=-self.mu, max=self.mu) # [rfn,pn]

        dist = (dist * dist_valid.float()).sum(0) / (dist_valid.float().sum(0) + 1e-6) # [pn]
        
        dist_all_invalid = (dist_valid.float().sum(0) == 0) # [pn]
        dist[dist_all_invalid] = 1e3
        
        outputs = {'dist': dist,
                   'valid_mask': ~dist_all_invalid}
        
        for k in return_names:
            inter_k = interpolate_feats(self.curr_obs_torch[k].permute(0,3,1,2),
                                        pts_2d,
                                        h = self.H,
                                        w = self.W,
                                        padding_mode='zeros',
                                        align_corners=True,
                                        inter_mode='bilinear') # [rfn,pn,k_dim]

            val = (inter_k * dist_valid.float().unsqueeze(-1) * dist_weight.unsqueeze(-1)).sum(0) / (dist_valid.float().sum(0).unsqueeze(-1) + 1e-6) # [pn,k_dim]
            val[dist_all_invalid] = 0.0
            
            outputs[k] = val
            if return_inter:
                outputs[k+'_inter'] = inter_k
            else:
                del inter_k

        return outputs

    def extract_dinov2_features(self, imgs, params):
        K, H, W, _ = imgs.shape
        
        patch_h = params['patch_h']
        patch_w = params['patch_w']
        feat_dim = 384 # vits14
        # feat_dim = 768 # vitb14
        # feat_dim = 1024 # vitl14
        # feat_dim = 1536 # vitg14
        
        transform = T.Compose([
            # T.GaussianBlur(9, sigma=(0.1, 2.0)),
            T.Resize((patch_h * 14, patch_w * 14)),
            T.CenterCrop((patch_h * 14, patch_w * 14)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        imgs_tensor = torch.zeros((K, 3, patch_h * 14, patch_w * 14), device=self.device)
        for j in range(K):
            img = Image.fromarray(imgs[j])
            imgs_tensor[j] = transform(img)[:3]
        with torch.no_grad():
            features_dict = self.dinov2_feat_extractor.forward_features(imgs_tensor.to(dtype=self.dtype))
            features = features_dict['x_norm_patchtokens']
            features = features.reshape((K, patch_h, patch_w, feat_dim))
        return features

    def extract_features(self, imgs, params):
        # :param imgs (K, H, W, 3) np array, color images
        # :param params: dict contains:
        #                - 'patch_h', 'patch_w': int, the size of the patch
        # :return features: (K, patch_h, patch_w, feat_dim) np array, features of the images

        if self.feat_backbone == 'dinov2':
            return self.extract_dinov2_features(imgs, params)
        else:
            raise NotImplementedError
    
    
    def update(self, obs):
        # :param obs: dict contains:
        #             - 'color': (K, H, W, 3) np array, color image
        #             - 'depth': (K, H, W) np array, depth image
        #             - 'pose': (K, 3, 4) np array, camera pose
        #             - 'K': (K, 3, 3) np array, camera intrinsics
        self.num_cam = obs['color'].shape[0]
        color = obs['color']
        params = {
            # 'patch_h': color.shape[1] // 10,
            # 'patch_w': color.shape[2] // 10,
            'patch_h': 64,
            'patch_w': 64,
        }
        # extract_dino_time = time.time()
        features = self.extract_features(color, params)

        self.curr_obs_torch['dino_feats'] = features
        self.curr_obs_torch['color'] = color
        self.curr_obs_torch['color_tensor'] = torch.from_numpy(color).to(self.device, dtype=self.dtype) / 255.0
        self.curr_obs_torch['depth'] = torch.from_numpy(obs['depth']).to(self.device, dtype=self.dtype)
        self.curr_obs_torch['pose'] = torch.from_numpy(obs['pose']).to(self.device, dtype=self.dtype)
        self.curr_obs_torch['K'] = torch.from_numpy(obs['K']).to(self.device, dtype=self.dtype)
        # self.curr_obs_torch['depth'] = obs['depth']
        # self.curr_obs_torch['pose'] = obs['pose']
        # self.curr_obs_torch['K'] = obs['K']
        _, self.H, self.W = obs['depth'].shape
        
    @torch.no_grad()
    def extract_semantic_feature_from_ptc(self, ptc, obs):
        self.update(obs)
        feature_dict = self.eval(ptc, return_names=['dino_feats'], return_inter=False)
        return feature_dict['dino_feats']
def read_pkl(file_path):
    with open(file_path, 'rb') as f:
        obs = pickle.load(f)
    return obs
def rgb_image_to_float_array(rgb_image, scale_factor=256000.0):
    """Recovers the depth values from an image.

    Reverses the depth to image conversion performed by FloatArrayToRgbImage or
    FloatArrayToGrayImage.

    The image is treated as an array of fixed point depth values.  Each
    value is converted to float and scaled by the inverse of the factor
    that was used to generate the Image object from depth values.  If
    scale_factor is specified, it should be the same value that was
    specified in the original conversion.

    The result of this function should be equal to the original input
    within the precision of the conversion.

    Args:
        image: Depth image output of FloatArrayTo[Format]Image.
        scale_factor: Fixed point scale factor.

    Returns:
        A 2D floating point numpy array representing a depth image.

    """
    image_array = np.array(rgb_image)
    image_shape = image_array.shape

    channels = image_shape[2] if len(image_shape) > 2 else 1
    assert 2 <= len(image_shape) <= 3
    if channels == 3:
        # RGB image needs to be converted to 24 bit integer.
        float_array = np.sum(image_array * [65536, 256, 1], axis=2)
    scaled_array = float_array / scale_factor
    return scaled_array

def transform_camera_extrinsics(extrinsics):
    # input extrinsics map camera to world coordinates (dim 3*4)
    # for example: C_world = R* C_cam + T
    # output extrinsics: map world coordinates to cam (dim 3*4)
    # C_cam = R* C_world + T
    C = np.expand_dims(extrinsics[:3, 3], 0).T
    R = extrinsics[:3, :3]
    R_inv = R.T
    R_inv_C = np.matmul(R_inv, C)
    output_extrinsics = np.concatenate((R_inv, -R_inv_C), -1) # 3*4
    return output_extrinsics

def process_episodes(begin, end, task, fusion, pcd_path,data_path,target_path,ptc_type,device):
    for episode in range(begin,end+1):
        pcd_folder = os.path.join(pcd_path, f'episode{episode}/{ptc_type}')
        camera_name=['over_shoulder_left','over_shoulder_right','overhead','wrist_right','wrist_left','front']
        episode_folder = os.path.join(data_path, f'episode{episode}')
        low_dim_obs_path = os.path.join(episode_folder, 'low_dim_obs.pkl')
        low_dim_obs = read_pkl(low_dim_obs_path)
        for i in range(len(low_dim_obs)):
            # start_time = time.time()
            pcd_step_folder = os.path.join(pcd_folder, f'step{i:03d}.npy')
            ptc = np.load(pcd_step_folder)
            #ptc = npy_array[i,:,:] # np.ndarray
            depth_image_lst = [] #cameraN, H, W
            cameras_extrinsics = []   #cameraN, 3,4
            cameras_intrinsics = []   #cameraN, 3,3
            all_cam_images = [] #cameraN, H,W,3 
            depth_folders = dict()
            cameras_near = dict()
            cameras_far = dict()
            #set_trace()
            for camera in camera_name:
                #set_trace()
                depth_folders[f'{camera}'] = (os.path.join(episode_folder, f'{camera}_depth'))
                cameras_extrinsics.append(transform_camera_extrinsics(low_dim_obs[i].misc[f'{camera}_camera_extrinsics'][:3,:]))
                cameras_intrinsics.append(low_dim_obs[i].misc[f'{camera}_camera_intrinsics'])
                cameras_near[f'{camera}'] = low_dim_obs[i].misc[f'{camera}_camera_near']
                cameras_far[f'{camera}'] = low_dim_obs[i].misc[f'{camera}_camera_far']
                if os.path.exists(depth_folders[f'{camera}']):
                    depth_images_path = os.path.join(depth_folders[f'{camera}'],f"depth_{i:04d}.png")
                    depth_image_rgb = np.array(Image.open(depth_images_path))
                    depth_image = rgb_image_to_float_array(depth_image_rgb, 2**24-1)
                    depth_image_m = (cameras_near[f'{camera}'] + depth_image * (cameras_far[f'{camera}'] - cameras_near[f'{camera}']))
                    depth_image_lst.append(depth_image_m)
                else:
                    print(f"Warning: {depth_folders[f'{camera}']} does not exist")
                camera_full_name = f"{camera}_rgb"
                image_path = os.path.join(episode_folder, camera_full_name, f"rgb_{i:04d}.png")
                image = np.array(Image.open(image_path))
                all_cam_images.append(image)
            all_cam_images = np.stack(all_cam_images,axis = 0)
            depth_image_lst = np.stack(depth_image_lst,axis=0)
            cameras_extrinsics = np.stack(cameras_extrinsics,axis=0)
            cameras_intrinsics = np.stack(cameras_intrinsics,axis=0)
            obs ={
                'color': all_cam_images,
                'depth': depth_image_lst,
                'pose': cameras_extrinsics,
                'K': cameras_intrinsics
            }

            ptc=torch.tensor(ptc[...,:3],dtype=torch.float32).to(device)
            # start_time=time.time()
            dino_feature_tensor = fusion.extract_semantic_feature_from_ptc(ptc,obs)
            dino_feature = dino_feature_tensor.cpu().numpy()
            print(dino_feature.shape)
            feature_folder = os.path.join(target_path, f'episode{episode}/{ptc_type}/step{i:03d}.npy')
            if not os.path.exists(os.path.dirname(feature_folder)):
                print(f"Creating directory: {os.path.dirname(feature_folder)}")
                os.makedirs(os.path.dirname(feature_folder))
            print(f"feature_folder{feature_folder}:{np.sum(np.all(dino_feature==0,axis=1))}")
            np.save(feature_folder, dino_feature)
        print(f"Episode {episode} finished!")
        
if __name__ == "__main__":
    device = 'cuda:1'
    task = 'bimanual_lift_ball'
    # task = 'bimanual_push_box'
    # task = 'bimanual_put_item_in_drawer'
    # task = 'bimanual_lift_tray'
    # task = 'bimanual_pick_laptop'
    # task = 'bimanual_handover_item_easy'
    # task = "bimanual_sweep_to_dustpan"
    # TODO
    pcd_path = f'data/training_processed/point_cloud/{task}/all_variations/episodes'
    data_path = f'data/training_raw/{task}/all_variations/episodes'
    target_path = f'data/training_processed/dino_feature/{task}/all_variations/episodes'
    ptc_type = 'rgb_pcd_rps6144'
    worker_threads = []
    for i in range(10):
        fusion = Fusion(num_cam=6,feat_backbone='dinov2',device=device)
        start = i * 10
        end = start + 9
        thread = threading.Thread(target=process_episodes, args=(start, end, task, fusion, pcd_path,data_path,target_path,ptc_type,device))
        worker_threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in worker_threads:
        thread.join()