import os
import pickle
from PIL import Image
import numpy as np
from pdb import set_trace
import threading
import imageio.v3 as iio
from segment_anything import SamPredictor, sam_model_registry
import torch
import cv2
import matplotlib.pyplot as plt
from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
import matplotlib.patches as patches
from pytorch3d.ops import sample_farthest_points
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, ax, color):
    ax.scatter(coords[:, 1], coords[:, 0], color=color, marker='*', s=1, edgecolor='white', linewidth=0.5)

class GetPointFlow():
    def __init__(self, data_path=None, target_path=None,task=None,device_num=None, text_prompt=None, prompt_type=None):
        self.data_path = data_path
        self.target_path = target_path
        # TODO: change path
        sam_checkpoint_path = "pretrained_models/sam_vit_b_01ec64.pth"
        device = f"cuda:{device_num}"
        sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint_path)
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)
        gdino_config_path = "repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
        gdino_checkpoint_path = "pretrained_models/groundingdino_swinb_cogcoor.pth"
        self.gdino_model = load_model(gdino_config_path, gdino_checkpoint_path)
        self.device=device
        self.task = task
        self.text_prompt=text_prompt
        self.prompt_type = prompt_type

    def read_pkl(self, file_path):
        with open(file_path, 'rb') as f:
            obs = pickle.load(f)
        return obs

    def read_pngs(self, folder_path):
        png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        png_files.sort()
        images = {}
        for png_file in png_files:
            img_path = os.path.join(folder_path, png_file)
            with Image.open(img_path) as img:
                images[png_file] = np.array(img)
        return images
    
    def rgb_image_to_float_array(self, rgb_image, scale_factor=256000.0):
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

    def _transform(self, coords, trans):
        # set_trace()
        h, w = coords.shape[:2]
        coords = np.reshape(coords, (h * w, -1))
        coords = np.transpose(coords, (1, 0))
        transformed_coords_vector = np.matmul(trans, coords)
        transformed_coords_vector = np.transpose(
            transformed_coords_vector, (1, 0))
        return np.reshape(transformed_coords_vector,
                        (h, w, -1))

    def _pixel_to_world_coords(self, pixel_coords, cam_proj_mat_inv):
        h, w = pixel_coords.shape[:2]
        pixel_coords = np.concatenate(
            [pixel_coords, np.ones((h, w, 1))], -1)
        world_coords = self._transform(pixel_coords, cam_proj_mat_inv)
        world_coords_homo = np.concatenate(
            [world_coords, np.ones((h, w, 1))], axis=-1)
        return world_coords_homo
    
    def project_points_coords(self, pts, Rt, K):
        """
        :param pts:  [pn,3]
        :param Rt:   [rfn,3,4]
        :param K:    [rfn,3,3]
        :return:
            coords:         [rfn,pn,2]
            invalid_mask:   [rfn,pn] 
            depth:          [rfn,pn,1]
        """
        # set_trace()
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
        return pts_2d

    def _create_uniform_pixel_coords_image(self, resolution: np.ndarray):
        '''
        Return: 
            uniform_pixel_coords: Shape (H, W, 3), each is (u, v, 1)
        '''
        pixel_x_coords = np.reshape(
            np.tile(np.arange(resolution[1]), [resolution[0]]),
            (resolution[0], resolution[1], 1)).astype(np.float32)
        pixel_y_coords = np.reshape(
            np.tile(np.arange(resolution[0]), [resolution[1]]),
            (resolution[1], resolution[0], 1)).astype(np.float32)
        pixel_y_coords = np.transpose(pixel_y_coords, (1, 0, 2))
        uniform_pixel_coords = np.concatenate(
            (pixel_x_coords, pixel_y_coords, np.ones_like(pixel_x_coords)), -1)
        return uniform_pixel_coords
    
    def pointcloud_from_depth_and_camera_params(
            self, 
            depth: np.ndarray, extrinsics: np.ndarray,
            intrinsics: np.ndarray) -> np.ndarray:
        """
        Converts depth (in meters) to point cloud in word frame.
        Return: A numpy array of size (width, height, 3)
        """
        upc = self._create_uniform_pixel_coords_image(depth.shape)
        pc = upc * np.expand_dims(depth, -1)
        C = np.expand_dims(extrinsics[:3, 3], 0).T
        R = extrinsics[:3, :3]
        R_inv = R.T
        R_inv_C = np.matmul(R_inv, C)
        extrinsics = np.concatenate((R_inv, -R_inv_C), -1) # 3*4
        cam_proj_mat = np.matmul(intrinsics, extrinsics) # 3*4
        cam_proj_mat_homo = np.concatenate(
            [cam_proj_mat, [np.array([0, 0, 0, 1])]]) # 4*4
        cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)[0:3]
        world_coords_homo = np.expand_dims(self._pixel_to_world_coords(
            pc, cam_proj_mat_inv), 0) # （1, H, W, 4）
        world_coords = world_coords_homo[..., :-1][0] # (H, W, 3)

        # set_trace()
        return world_coords
    
    def sample_points(self, sam_prompt, camera, episode):
        episode_folder = os.path.join(self.data_path, f'episode{episode}')
        img_path = os.path.join(episode_folder, f'{camera}_rgb', "rgb_0000.png")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor = self.sam_predictor
        predictor.set_image(image)
        input_label = np.array([1])
        if self.prompt_type=="box":
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=sam_prompt,
                multimask_output=False,
            )
            mask = masks[0].cpu().numpy()
        else:
            mask, score, logit = predictor.predict(
                point_coords=sam_prompt.reshape(1,2),
                point_labels=input_label,
                multimask_output=False,
            )
        # set_trace()
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1)
        true_coords = np.argwhere(mask_image[:, :, 0])
        return true_coords

    def predict_tracks(self, episode, camera):
        # set_trace()
        sam_prompt = self.get_sam_prompt_point_from_foundationdino(episode,camera)
        queries = self.sample_points(sam_prompt, camera, episode)
        return queries
    
    def pointflow_from_tracks(
            self, 
            depth, extrinsics,
            intrinsics, pixel_tracks):
        """
        pixel_tracks: Npts, 2
        depth:256,256
        Converts depth (in meters) to point cloud in word frame.
        Return: A numpy array of size (width, height, 3)
        """
        # set_trace()
        Npts,_ = pixel_tracks.shape
        pixel_tracks = pixel_tracks.astype(int)
        ones_column = np.ones((Npts,1))
        upc = np.hstack((pixel_tracks[:,[1,0]],ones_column))
        # set_trace()
        depth_values = depth[pixel_tracks[:,0],pixel_tracks[:,1]].reshape(-1,1)
        pc = upc * depth_values
        #pc: Npts,3
        C = np.expand_dims(extrinsics[:3, 3], 0).T
        R = extrinsics[:3, :3]
        R_inv = R.T
        R_inv_C = np.matmul(R_inv, C)
        extrinsics = np.concatenate((R_inv, -R_inv_C), -1) # 3*4
        cam_proj_mat = np.matmul(intrinsics, extrinsics) # 3*4
        cam_proj_mat_homo = np.concatenate(
            [cam_proj_mat, [np.array([0, 0, 0, 1])]]) # 4*4
        cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)[0:3]
        world_coords_homo = self._2dpixel_to_world_coords(pc, cam_proj_mat_inv)
        world_coords = world_coords_homo[..., :-1] # (Npts, 3)

        # set_trace()
        return world_coords
    
    def _2dtransform(self, coords, trans):
        Npts = coords.shape[0]
        coords = np.transpose(coords, (1, 0))
        transformed_coords_vector = np.matmul(trans, coords)
        transformed_coords_vector = np.transpose(
            transformed_coords_vector, (1, 0))
        return np.reshape(transformed_coords_vector,
                        (Npts, -1))

    def _2dpixel_to_world_coords(self, pixel_coords, cam_proj_mat_inv):
        # set_trace()
        Npts = pixel_coords.shape[0]
        pixel_coords = np.concatenate(
            [pixel_coords, np.ones((Npts, 1))], -1)
        world_coords = self._2dtransform(pixel_coords, cam_proj_mat_inv)
        world_coords_homo = np.concatenate(
            [world_coords, np.ones((Npts, 1))], axis=-1)
        return world_coords_homo
    
        
    def get_sam_prompt_point(self, episode, camera):
        # set_trace()
        episode_folder = os.path.join(self.data_path, f'episode{episode}')
        low_dim_obs_path = os.path.join(episode_folder, 'low_dim_obs.pkl')
        low_dim_obs = self.read_pkl(low_dim_obs_path)
        object_position = low_dim_obs[0].object_6d_pose['position']
        object_position = torch.tensor(object_position,dtype=torch.float32).unsqueeze(0).to(self.device)
        # 1,3
        pose=self.transform_camera_extrinsics(low_dim_obs[0].misc[f'{camera}_camera_extrinsics'][:3,:])
        K=low_dim_obs[0].misc[f'{camera}_camera_intrinsics']
        pose = torch.tensor(pose,dtype=torch.float32).unsqueeze(0).to(self.device)
        K = torch.tensor(K,dtype=torch.float32).unsqueeze(0).to(self.device)
        object_2d_projection = self.project_points_coords(object_position, pose, K).squeeze(1)
        return object_2d_projection
    
    def get_sam_prompt_point_from_foundationdino(self, episode, camera):
        device = self.device
        image_path = os.path.join(self.data_path, f'episode{episode}', f'{camera}_rgb', "rgb_0000.png")
        image_source, image = load_image(image_path)
        # text_prompt = "a white ball"
        boxes, logits, phrases = predict(model=self.gdino_model,
                                         image=image,
                                         caption=self.text_prompt,
                                         box_threshold=0.35,
                                         text_threshold=0.35,
                                         device=device)
        boxes = boxes[0].reshape(1,4)
        H, W, _ = image_source.shape
        boxes_np = boxes.cpu().numpy() * np.array([W, H, W, H])
        cx, cy, wx, wy = boxes_np[:, 0], boxes_np[:, 1], boxes_np[:, 2], boxes_np[:, 3]
        prompt_boxes =  np.array([cx - wx/2, cy - wy/2, cx + wx/2, cy + wy/2]).T
        prompt_boxes = prompt_boxes.astype(int)
        input_boxes = torch.tensor(prompt_boxes[0]).to(self.device)
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(input_boxes, image_source.shape[:2])
        prompt_point = np.array([cx[0], cy[0]])
        if self.prompt_type=="box":
            return transformed_boxes
        else:
            return prompt_point.astype(int)
        
        
    def transform_camera_extrinsics(self,extrinsics):
        C = np.expand_dims(extrinsics[:3, 3], 0).T
        R = extrinsics[:3, :3]
        R_inv = R.T
        R_inv_C = np.matmul(R_inv, C)
        output_extrinsics = np.concatenate((R_inv, -R_inv_C), -1)
        return output_extrinsics
    
    def world_to_object_coordinates(self, world_points, matrix):
        R = matrix[:3,:3]
        t = np.expand_dims(matrix[:3, 3], 0)
        R_inv = R.T
        points_in_object_frame = np.matmul(world_points - t, R_inv.T)
        return points_in_object_frame
    
    def object_to_world_coordinates(self, object_points, matrix):
        R = matrix[:3,:3]
        t = np.expand_dims(matrix[:3, 3], 0)
        R_inv = R.T
        points_in_world_frame = np.matmul(object_points,R.T) + t
        return points_in_world_frame
    
    
    def process_episodes(self, start, end, cameras, pcd_type):
        # set_trace()
        cameras_extrinsics = dict()
        cameras_intrinsics = dict()
        cameras_near = dict()
        cameras_far = dict()

        for episode in range(start, end + 1):
            thirdview_point_cloud = None
            
            episode_folder = os.path.join(self.data_path, f'episode{episode}')
            low_dim_obs_path = os.path.join(episode_folder, 'low_dim_obs.pkl')
            lang_path = os.path.join(episode_folder, 'variation_descriptions.pkl')
            # if int(os.environ["LOCAL_RANK"]) == 0:
            # print(f'lang_path: {lang_path}')
            if os.path.exists(lang_path):
                lang = self.read_pkl(lang_path)[0]
                # color_name = lang.split()[3]
                # self.text_prompt = f'a small {color_name} cube'
                # self.text_prompt = f'green cube'

                # print(lang)
                print(self.text_prompt)
            # set_trace()
            depth_folders = dict()
            rgb_folders = dict()
            
            if os.path.exists(low_dim_obs_path):
                low_dim_obs = self.read_pkl(low_dim_obs_path)
                for camera in cameras:
                    depth_folders[camera] = (os.path.join(episode_folder, f'{camera}_depth')) # front / overhead
                    if 'rgb' in pcd_type:
                        rgb_folders[camera] = (os.path.join(episode_folder, f'{camera}_rgb')) # front / overhead
                    cameras_extrinsics[camera] = low_dim_obs[0].misc[f'{camera}_camera_extrinsics']
                    cameras_intrinsics[camera] = low_dim_obs[0].misc[f'{camera}_camera_intrinsics']
                    cameras_near[camera] = low_dim_obs[0].misc[f'{camera}_camera_near']
                    cameras_far[camera] = low_dim_obs[0].misc[f'{camera}_camera_far']
            else:
                print(f"Warning: {low_dim_obs_path} does not exist")

            
            point_cloud = dict()
            for camera in cameras:
                if os.path.exists(depth_folders[camera]):
                    # depth_images = self.read_pngs(depth_folders[camera])
                    episode_tracks = self.predict_tracks(episode, camera)   #Npts, 2
                    depth_file0 = os.path.join(depth_folders[camera], "depth_0000.png")
                    with Image.open(depth_file0) as img:
                        depth_image_rgb = np.array(img)
                    depth_image = self.rgb_image_to_float_array(depth_image_rgb, 2**24-1)
                    depth_image_m = (cameras_near[camera] + depth_image * (cameras_far[camera] - cameras_near[camera]))
                    
                    pc_initial = self.pointflow_from_tracks(
                            depth_image_m, cameras_extrinsics[camera],
                            cameras_intrinsics[camera], episode_tracks)
                    # set_trace()
                    step_num = len(low_dim_obs)
                    object_pose_initial = low_dim_obs[0].object_6d_pose["matrix"]
                    pc_object_frame = self.world_to_object_coordinates(pc_initial, object_pose_initial)

                    point_cloud[camera] = [pc_initial]
                    for step in range(1, step_num):
                        object_pose_t = low_dim_obs[step].object_6d_pose["matrix"]
                        pc = self.object_to_world_coordinates(pc_object_frame, object_pose_t)
                        point_cloud[camera].append(pc) # T, 65536, 3
                    # set_trace()
                    if thirdview_point_cloud is None:
                        thirdview_point_cloud = point_cloud[camera]
                    else:
                        thirdview_point_cloud = np.concatenate([thirdview_point_cloud, point_cloud[camera]], axis=1) # T, 65536 * camera_num, 3
                else:
                    print(f"Warning: {depth_folders[camera]} does not exist")


            # set_trace()
            thirdview_point_cloud = np.array(thirdview_point_cloud)
            thirdview_point_cloud = torch.tensor(thirdview_point_cloud)
            
            T, total_points, _ = thirdview_point_cloud.shape
            # set_trace()
            ps_num = int(pcd_type.split('ps')[-1])
            rand_idx = np.random.choice(total_points, int(ps_num), replace=False)
            rand_idx = sorted(rand_idx)
            
            sampled_pc = thirdview_point_cloud[:,rand_idx]
            for t in range(T):
                pcd_folder = os.path.join(self.target_path, f'episode{episode}/{pcd_type}/step{t:03d}.npy')
                if not os.path.exists(os.path.dirname(pcd_folder)):
                    os.makedirs(os.path.dirname(pcd_folder))
                print('pcd_folder: ', pcd_folder)
                print(sampled_pc[t].shape)
                np.save(pcd_folder, sampled_pc[t])
                # set_trace()

        return sampled_pc
    
if __name__ == "__main__":
    # TODO: change task
    # task = 'bimanual_handover_item_easy' 
    task = 'bimanual_lift_ball' 
    # task = 'bimanual_lift_tray' 
    # task = 'bimanual_pick_laptop'  
    # task = 'bimanual_push_box' 
    # task = 'bimanual_put_item_in_drawer' 
    # task = 'bimanual_sweep_to_dustpan' 

    print(task)
    # TODO: change data_path and target_path
    data_path = f'data/training_raw/{task}/all_variations/episodes'
    target_path = f'data/training_processed/point_flow/{task}/all_variations/episodes'
    
    pcd_type = 'world_ordered_rps200'

    # [IMPORTANT!!] TODO: change prompts and cameras for different tasks
    # ball
    text_prompt = 'a white ball'
    cameras = ['over_shoulder_right']
    prompt_type = "box"


    # box: 
    # text_prompt = 'a white rectangle box'
    # cameras = ['over_shoulder_right']
    # prompt_type = "box"

    # handover_easy: 
    # text_prompt = 'a small red cube'
    # cameras = ['front']
    # prompt_type = "box"
    
    # lift tray
    # cameras = ['front']
    # text_prompt = 'a grey tray.'
    # prompt_type = "box"
    
    # bimanual_put_item_in_drawer
    # cameras=["over_shoulder_left"]
    # text_prompt = "a small white cube."
    # prompt_type = "box"
    
    # bimanual_sweep_to_dustpan
    # cameras=["over_shoulder_left"]
    # text_prompt = "a top-down view of a brown T-shaped pole and its bottom."
    # prompt_type = "box"
    
    # laptop 
    # cameras=["front"]
    # text_prompt = "a black rectangle laptop"
    # prompt_type = "point"
    
    print(cameras)
    print(pcd_type)
   
    get_point_flow = GetPointFlow(data_path, target_path,task,0,text_prompt, prompt_type)
    get_point_flow.process_episodes(0, 99, cameras, pcd_type)