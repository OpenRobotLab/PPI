import os
import pickle
from PIL import Image
import numpy as np
import zarr
from pdb import set_trace
import threading
from cotracker.utils.visualizer import Visualizer
import imageio.v3 as iio
from segment_anything import SamPredictor, sam_model_registry
import torch
import cv2
import matplotlib.pyplot as plt
from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
import matplotlib.patches as patches
from pytorch3d.ops import sample_farthest_points
import glob

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

class GetPointFlow_multi_objs():
    def __init__(self, data_path=None, target_path=None,task=None,device_num=None, text_prompt=None, prompt_type=None, objs_and_pfl_num=None):
        self.data_path = data_path
        self.target_path = target_path
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
        self.objs_and_pfl_num = objs_and_pfl_num

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
    
    def sample_points(self, sam_prompts, camera, episode):
        episode_folder = os.path.join(self.data_path, f'episode{episode:04d}')
        img_path = f"{episode_folder}/steps/0000/{camera}_rgb.jpg"
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor = self.sam_predictor

        true_coords_multi = {}

        objs_name_list = list(self.objs_and_pfl_num.keys())

        for i, sam_prompt in enumerate(sam_prompts):
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

            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1)
            true_coords = np.argwhere(mask_image[:, :, 0]) 

            true_coords_multi[objs_name_list[i]] = true_coords
         
        return true_coords_multi
        
    def predict_tracks(self, episode, camera):
        # set_trace()
        sam_prompts = self.get_sam_prompt_point_from_foundationdino(episode, camera)
        print(sam_prompts)

        queries_multi = self.sample_points(sam_prompts, camera, episode)

        return queries_multi
    
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
        # Npts = coords.shape[0]
        # coords = np.reshape(coords, (h * w, -1))
        # set_trace()
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
        image_path = f"{self.data_path}/episode{episode:04d}/steps/0000/{camera}_rgb.jpg"
        image_source, image = load_image(image_path)
        transformed_boxes = []
        for text_prompt in self.text_prompt:
            box, logits, phrases = predict(model=self.gdino_model,
                                            image=image,
                                            caption=text_prompt,
                                            box_threshold=0.35,
                                            text_threshold=0.8,
                                            device=device)

            box = box[0].reshape(1,4)
            H, W, _ = image_source.shape
            box_np = box.cpu().numpy() * np.array([W, H, W, H])
            cx, cy, wx, wy = box_np[:, 0], box_np[:, 1], box_np[:, 2], box_np[:, 3]
            prompt_box =  np.array([cx - wx/2, cy - wy/2, cx + wx/2, cy + wy/2]).T
            prompt_box = prompt_box.astype(int)
            input_box = torch.tensor(prompt_box[0]).to(self.device)
            transformed_box = self.sam_predictor.transform.apply_boxes_torch(input_box, image_source.shape[:2])
            prompt_point = np.array([cx[0], cy[0]])

            transformed_boxes.append(transformed_box)

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

        for episode in range(start, end + 1):
            thirdview_point_cloud = None
            
            episode_folder = os.path.join(self.data_path, f'episode{episode:04d}')

            steps_dir = f"{data_path}/episode{episode:04d}/steps"
            steps = sorted(glob.glob(f'{steps_dir}/*'))
            steps_num = len(steps)

            other_data_path = f"{steps[0]}/other_data.pkl"

            if os.path.exists(other_data_path):
                other_data = self.read_pkl(other_data_path)
                for camera in cameras:
                    cameras_extrinsics[camera] = other_data['extr'][camera]
                    cameras_intrinsics[camera] = other_data['intr'][camera]
            else:
                print(f"Warning: {other_data_path} does not exist")

            
            point_cloud = dict()
            for camera in cameras:
                episode_tracks_multi = self.predict_tracks(episode, camera)   # Npts, 2
                
                depth_file0 = f"{episode_folder}/steps/0000/{camera}_depth_x10000_uint16.png"
                with Image.open(depth_file0) as img:
                    depth_image = np.array(img)
                depth_image_m = depth_image / 10000
                pc_initial_multi = {}
                #####################################################
                for obj_name, episode_tracks in episode_tracks_multi.items():
                    pc_initial_current = self.pointflow_from_tracks(
                            depth_image_m, cameras_extrinsics[camera],
                            cameras_intrinsics[camera], episode_tracks) 

                    mask = (pc_initial_current[:, 0] >= 0) & (pc_initial_current[:, 0] <= 1) 
                    
                    pc_initial_multi[obj_name] = pc_initial_current[mask]

                    # ps_num = int(pcd_type.split('ps')[-1])
                    ps_num = self.objs_and_pfl_num[obj_name]
                    total_points = pc_initial_multi[obj_name].shape[0]
                    rand_idx = np.random.choice(total_points, ps_num, replace=False)
                    rand_idx = sorted(rand_idx)
                    pc_initial_multi[obj_name] = pc_initial_multi[obj_name][rand_idx]

                # set_trace()
                obj_6dposes = {}
                pc_object_frame_multi = {}
                for obj_name, pc_initial in pc_initial_multi.items():
                    obj_6dpose_path = f"{self.data_path}/obj_6dpose/episode{episode:04d}/obj_6dpose_{obj_name}.pkl"
                    obj_6dposes[obj_name] = self.read_pkl(obj_6dpose_path)
                    object_pose_initial = obj_6dposes[obj_name][0]
                    pc_object_frame_multi[obj_name] = self.world_to_object_coordinates(pc_initial, object_pose_initial) 

                pc_initial_merged = np.concatenate(list(pc_initial_multi.values()), axis=0)
                point_cloud[camera] = [pc_initial_merged]

                for step in range(1, steps_num):
                    pc_current_multi = {}
                    for obj_name, pc_object_frame in pc_object_frame_multi.items():
                        object_pose_t = obj_6dposes[obj_name][step]
                        pc_current_multi[obj_name] = self.object_to_world_coordinates(pc_object_frame, object_pose_t)
                    
                    pc_current_merged = np.concatenate(list(pc_current_multi.values()), axis=0)
                    point_cloud[camera].append(pc_current_merged) # T, 200, 3

                if thirdview_point_cloud is None:
                    thirdview_point_cloud = point_cloud[camera]
                else:
                    thirdview_point_cloud = np.concatenate([thirdview_point_cloud, point_cloud[camera]], axis=1) # T, 200 * camera_num, 3

            thirdview_point_cloud = np.array(thirdview_point_cloud)
            ###################################################################
            T, total_points, _ = thirdview_point_cloud.shape
            # set_trace()
            
            for t in range(T):
                pcd_folder = os.path.join(self.target_path, f'episode{episode}/{pcd_type}/step{t:03d}.npy')
                if not os.path.exists(os.path.dirname(pcd_folder)):
                    os.makedirs(os.path.dirname(pcd_folder))
                print('pcd_folder: ', pcd_folder)
                print(thirdview_point_cloud[t].shape)
                np.save(pcd_folder, thirdview_point_cloud[t])

        return None

if __name__ == "__main__":
    # TODO: change task
    # task = "handover_and_insert_the_plate"
    task = "wipe_the_plate" 
    # task = "press_the_bottle"
    # task = "scan_the_bottle"
    # task = 'wear_the_scarf' 
    
    print(task)
    data_path = f'real_data/training_raw/{task}'
    # TODO: change the path, 'training_processed' for normal setting, 'training_processed_simple' for simple setting
    target_path = f'real_data/training_processed/point_flow/{task}'
    cameras=['head']   
    # TODO: 200 for normal setting, 50 for simple setting
    pcd_type = 'world_ordered_rps200'
    print(cameras)
    print(pcd_type)

    # TODO: change text_prompt and objs_and_pfl_num
    text_prompt = ["A sponge.", "A purple plate."]
    objs_and_pfl_num = {"sponge": 50, "plate": 150}

    print(text_prompt)

    prompt_type = "box"
    get_point_flow = GetPointFlow_multi_objs(data_path, target_path, task, 0, text_prompt, prompt_type, objs_and_pfl_num)
    # TODO: change the episode number
    demo_num = 20
    get_point_flow.process_episodes(0, demo_num-1, cameras, pcd_type)
