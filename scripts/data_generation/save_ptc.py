import os
import pickle
from PIL import Image
import numpy as np
from pdb import set_trace
import threading

class pcd():
    def __init__(self, data_path=None, target_path=None):
        self.data_path = data_path
        self.target_path = target_path

    def read_pkl(self, file_path):
        with open(file_path, 'rb') as f:
            obs = pickle.load(f)
        return obs
    
    def FPS(self, pc, num_samples):
        """
        Perform farthest point sampling (FPS) on the point cloud.
        
        Args:
            pc: numpy array of shape (N, 3), where N is the number of points.
            num_samples: Number of points to sample.
            
        Returns:
            sampled_pc: numpy array of shape (num_samples, 3), sampled points.
        """
        if num_samples >= pc.shape[0]:
            return pc
        
        sampled_indices = np.zeros(num_samples, dtype=int)
        distances = np.full(pc.shape[0], np.inf)
        
        # Start with a random point
        sampled_indices[0] = np.random.randint(0, pc.shape[0])
        
        for i in range(1, num_samples):
            # Compute distance from all points to the nearest sampled point
            dists = np.linalg.norm(pc - pc[sampled_indices[i-1]], axis=1)
            distances = np.minimum(distances, dists)
            
            # Select the farthest point
            sampled_indices[i] = np.argmax(distances)
        
        return pc[sampled_indices]

    def FPS_with_bounding_box(self, pc, num_samples, bounding_box):
        """
        Perform farthest point sampling (FPS) on the point cloud within a bounding box.
        
        Args:
            pc: numpy array of shape (N, 3), where N is the number of points.
            num_samples: Number of points to sample.
            bounding_box: numpy array of shape (2, 3), where the first row contains the
                        minimum x, y, z values and the second row contains the
                        maximum x, y, z values of the bounding box.
        
        Returns:
            sampled_pc: numpy array of shape (num_samples, 3), sampled points within the bounding box.

        Example usage:
            pc = np.random.rand(1000, 3)  # Generate a random point cloud
            bounding_box = np.array([[0, 0, 0], [1, 1, 1]])  # Define a bounding box
            sampled_points = self.FPS_with_bounding_box(pc, 10, bounding_box)
        """
        if num_samples >= pc.shape[0]:
            return pc
        
        # Extract the min and max corners of the bounding box
        min_corner = bounding_box[0]
        max_corner = bounding_box[1]
        
        # Filter points to only include those within the bounding box
        inside_mask = (pc >= min_corner) & (pc <= max_corner)
        inside_mask = inside_mask[:, 0] & inside_mask[:, 1] & inside_mask[:, 2]
        inside_pc = pc[inside_mask]
        
        if inside_pc.shape[0] < num_samples:
            raise ValueError("Not enough points inside the bounding box to sample.")
        
        sampled_indices = np.zeros(num_samples, dtype=int)
        distances = np.full(inside_pc.shape[0], np.inf)
        
        # Start with a random point
        sampled_indices[0] = np.random.randint(0, inside_pc.shape[0])
        
        for i in range(1, num_samples):
            # Compute distance from all points to the nearest sampled point
            dists = np.linalg.norm(inside_pc - inside_pc[sampled_indices[i-1]], axis=1)
            distances = np.minimum(distances, dists)
            
            # Select the farthest point
            sampled_indices[i] = np.argmax(distances)
        
        return inside_pc[sampled_indices]

    def run_fps(self, context_features, context_pos):
        # context_features (Np, B, F)
        # context_pos (B, Np, F, 2)
        # outputs of analogous shape, with smaller Np
        npts, bs, ch = context_features.shape

        # Sample points with FPS
        sampled_inds = dgl_geo.farthest_point_sampler(
            einops.rearrange(
                context_features,
                "npts b c -> b npts c"
            ).to(torch.float64),
            max(npts // self.fps_subsampling_factor, 1), 0
        ).long()

        # Sample features
        expanded_sampled_inds = sampled_inds.unsqueeze(-1).expand(-1, -1, ch)
        sampled_context_features = torch.gather(
            context_features,
            0,
            einops.rearrange(expanded_sampled_inds, "b npts c -> npts b c")
        )
        return sampled_context_features
    def RPS_with_bounding_box(self, pc, num_samples, bounding_box):
      
        # Extract the min and max corners of the bounding box
        min_corner = bounding_box[0]
        max_corner = bounding_box[1]
        
        pc_xyz = pc[:, :3]
        # print(f'pc_xyz.shape: {pc_xyz.shape}')
        # Filter points to only include those within the bounding box
        inside_mask = (pc_xyz >= min_corner) & (pc_xyz <= max_corner)
        inside_mask = inside_mask[:, 0] & inside_mask[:, 1] & inside_mask[:, 2]
        inside_pc = pc[inside_mask]

        rand_idx = np.random.choice(inside_pc.shape[0], int(num_samples), replace=False)
        return inside_pc[rand_idx]

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
        h, w = coords.shape[:2]
        coords = np.reshape(coords, (h * w, -1))
        coords = np.transpose(coords, (1, 0))
        transformed_coords_vector = np.matmul(trans, coords)
        transformed_coords_vector = np.transpose(
            transformed_coords_vector, (1, 0))
        return np.reshape(transformed_coords_vector,
                        (h, w, -1))

    def _pixel_to_world_coords(self, pixel_coords, cam_proj_mat_inv):
        # set_trace()
        h, w = pixel_coords.shape[:2]
        pixel_coords = np.concatenate(
            [pixel_coords, np.ones((h, w, 1))], -1)
        world_coords = self._transform(pixel_coords, cam_proj_mat_inv)
        world_coords_homo = np.concatenate(
            [world_coords, np.ones((h, w, 1))], axis=-1)
        return world_coords_homo

    def _create_uniform_pixel_coords_image(self, resolution: np.ndarray):
        # set_trace()
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
        # set_trace()
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


        return world_coords

    def process_episodes(self, start, end, cameras, pcd_type, bounding_box):
        cameras_extrinsics = dict()
        cameras_intrinsics = dict()
        cameras_near = dict()
        cameras_far = dict()

        for episode in range(start, end + 1):
            thirdview_point_cloud = None
            wrist_point_cloud = None
            
            episode_folder = os.path.join(self.data_path, f'episode{episode}')
            low_dim_obs_path = os.path.join(episode_folder, 'low_dim_obs.pkl')
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
                    depth_images = self.read_pngs(depth_folders[camera])
                    if 'rgb' in pcd_type:
                        rgb_images = self.read_pngs(rgb_folders[camera])

                    for depth_file, depth_image_rgb in depth_images.items():
                        rgb_file = 'rgb_' + depth_file.split('_')[-1]
                        rgb_image = rgb_images.get(rgb_file) 

                        depth_image = self.rgb_image_to_float_array(depth_image_rgb, 2**24-1)
                        depth_image_m = (cameras_near[camera] + depth_image * (cameras_far[camera] - cameras_near[camera]))
                        step = int(depth_file.split('_')[1][:4])
                        # print(f"step:{step}")
                        # print(f"low_dim_obs{len(low_dim_obs)}")
                        cameras_extrinsics[camera] = low_dim_obs[step].misc[f'{camera}_camera_extrinsics']
                        pc = self.pointcloud_from_depth_and_camera_params(
                        depth_image_m,
                        cameras_extrinsics[camera],
                        cameras_intrinsics[camera]
                        ).reshape(-1, 3) # 65536, 3

                        if 'rgb' in pcd_type:
                            rgb = rgb_image.reshape(-1, 3)
                            rgb_pc = np.concatenate([pc, rgb], axis=-1)

                        if camera not in point_cloud:
                            point_cloud[camera] = [] 
                            if 'rgb' in pcd_type:
                                point_cloud[camera].append(rgb_pc) # T, 65536, 3
                            else:
                                point_cloud[camera].append(pc) # T, 65536, 3
                        else:
                            if 'rgb' in pcd_type:
                                point_cloud[camera].append(rgb_pc) # T, 65536, 3
                            else:
                                point_cloud[camera].append(pc) # T, 65536, 3

                    if thirdview_point_cloud is None:
                        thirdview_point_cloud = point_cloud[camera]
                    else:
                        thirdview_point_cloud = np.concatenate([thirdview_point_cloud, point_cloud[camera]], axis=1) # T, 65536 * camera_num, 3
                else:
                    print(f"Warning: {depth_folders[camera]} does not exist")

            T, _, _ = thirdview_point_cloud.shape
            ps_num = int(pcd_type.split('ps')[-1])
            
            for t in range(T):
                pcd_folder = os.path.join(self.target_path, f'episode{episode}/{pcd_type}/step{t:03d}.npy')
                if not os.path.exists(os.path.dirname(pcd_folder)):
                    os.makedirs(os.path.dirname(pcd_folder))
                print('pcd_folder: ', pcd_folder)
                thirdview_point_cloud_ps = self.RPS_with_bounding_box(thirdview_point_cloud[t], ps_num , bounding_box)

                print(thirdview_point_cloud_ps.shape)
                np.save(pcd_folder, thirdview_point_cloud_ps)

        return thirdview_point_cloud_ps

    def process_episodes_weightedpc(self, start, end, cameras, pcd_type, bounding_box):
        cameras_extrinsics = dict()
        cameras_intrinsics = dict()
        cameras_near = dict()
        cameras_far = dict()

        for episode in range(start, end + 1):
            thirdview_point_cloud = None
            wrist_point_cloud = None
            
            episode_folder = os.path.join(self.data_path, f'episode{episode}')
            low_dim_obs_path = os.path.join(episode_folder, 'low_dim_obs.pkl')
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
                if camera in ['wrist_left', 'wrist_right']:
                    continue

                if os.path.exists(depth_folders[camera]):
                    depth_images = self.read_pngs(depth_folders[camera])
                    if 'rgb' in pcd_type:
                        rgb_images = self.read_pngs(rgb_folders[camera])

                    for depth_file, depth_image_rgb in depth_images.items():
                        rgb_file = 'rgb_' + depth_file.split('_')[-1]
                        rgb_image = rgb_images.get(rgb_file) 

                        depth_image = self.rgb_image_to_float_array(depth_image_rgb, 2**24-1)
                        depth_image_m = (cameras_near[camera] + depth_image * (cameras_far[camera] - cameras_near[camera]))
                        
                        pc = self.pointcloud_from_depth_and_camera_params(
                        depth_image_m,
                        cameras_extrinsics[camera],
                        cameras_intrinsics[camera]
                        ).reshape(-1, 3) # 65536, 3

                        if 'rgb' in pcd_type:
                            rgb = rgb_image.reshape(-1, 3)
                            rgb_pc = np.concatenate([pc, rgb], axis=-1)

                        if camera not in point_cloud:
                            point_cloud[camera] = [] 
                            if 'rgb' in pcd_type:
                                point_cloud[camera].append(rgb_pc) # T, 65536, 3
                            else:
                                point_cloud[camera].append(pc) # T, 65536, 3
                        else:
                            if 'rgb' in pcd_type:
                                point_cloud[camera].append(rgb_pc) # T, 65536, 3
                            else:
                                point_cloud[camera].append(pc) # T, 65536, 3

                    if thirdview_point_cloud is None:
                        thirdview_point_cloud = point_cloud[camera]
                    else:
                        thirdview_point_cloud = np.concatenate([thirdview_point_cloud, point_cloud[camera]], axis=1) # T, 65536 * camera_num, 3
                else:
                    print(f"Warning: {depth_folders[camera]} does not exist")

            for camera in ['wrist_left', 'wrist_right']:
                if os.path.exists(depth_folders[camera]):
                    depth_images = self.read_pngs(depth_folders[camera])
                    if 'rgb' in pcd_type:
                        rgb_images = self.read_pngs(rgb_folders[camera])
                        
                    for depth_file, depth_image_rgb in depth_images.items():
                        rgb_file = 'rgb_' + depth_file.split('_')[-1]
                        rgb_image = rgb_images.get(rgb_file) 

                        step = int(depth_file.split('_')[1][:4])
                        cameras_extrinsics[camera] = low_dim_obs[step].misc[f'{camera}_camera_extrinsics']

                        depth_image = self.rgb_image_to_float_array(depth_image_rgb, 2**24-1)
                        depth_image_m = (cameras_near[camera] + depth_image * (cameras_far[camera] - cameras_near[camera]))
                        
                        pc = self.pointcloud_from_depth_and_camera_params(
                        depth_image_m,
                        cameras_extrinsics[camera],
                        cameras_intrinsics[camera]
                        ).reshape(-1, 3) # 65536, 3

                        if 'rgb' in pcd_type:
                            rgb = rgb_image.reshape(-1, 3)
                            rgb_pc = np.concatenate([pc, rgb], axis=-1)

                        if camera not in point_cloud:
                            point_cloud[camera] = [] 
                            if 'rgb' in pcd_type:
                                point_cloud[camera].append(rgb_pc) # T, 65536, 3
                            else:
                                point_cloud[camera].append(pc) # T, 65536, 3
                        else:
                            if 'rgb' in pcd_type:
                                point_cloud[camera].append(rgb_pc) # T, 65536, 3
                            else:
                                point_cloud[camera].append(pc) # T, 65536, 3

                    if wrist_point_cloud is None:
                        wrist_point_cloud = point_cloud[camera]
                    else:
                        wrist_point_cloud = np.concatenate([wrist_point_cloud, point_cloud[camera]], axis=1) # T, 65536 * camera_num, 3
                else:
                    print(f"Warning: {depth_folders[camera]} does not exist")


            T, _, _ = thirdview_point_cloud.shape
            ps_num = int(pcd_type.split('ps')[-1])
            
            for t in range(T):
                pcd_folder = os.path.join(self.target_path, f'episode{episode}/{pcd_type}/step{t:03d}.npy')
                if not os.path.exists(os.path.dirname(pcd_folder)):
                    os.makedirs(os.path.dirname(pcd_folder))
                print('pcd_folder: ', pcd_folder)
                thirdview_point_cloud_ps = self.RPS_with_bounding_box(thirdview_point_cloud[t], ps_num / 6 * 4 , bounding_box)
                wrist_point_cloud_ps = self.RPS_with_bounding_box(wrist_point_cloud[t], ps_num / 6 * 2 , bounding_box)
                multiview_point_cloud_ps = np.concatenate([thirdview_point_cloud_ps, wrist_point_cloud_ps], axis=0)

                print(multiview_point_cloud_ps.shape)
                np.save(pcd_folder, multiview_point_cloud_ps)

        return multiview_point_cloud_ps


if __name__ == "__main__":
    # TODO
    # task = 'bimanual_handover_item_easy' 
    task = 'bimanual_lift_ball' 
    # task = 'bimanual_lift_tray' 
    # task = 'bimanual_pick_laptop' 
    # task = 'bimanual_push_box' 
    # task = 'bimanual_put_item_in_drawer' 
    # task = 'bimanual_sweep_to_dustpan' 

    print(task)
    # TODO
    data_path = f'data/training_raw/{task}/all_variations/episodes'
    target_path = f'data/training_processed/point_cloud/{task}/all_variations/episodes'
    getpcd = pcd(data_path, target_path)
    cameras = ['over_shoulder_left', 'over_shoulder_right', 'overhead', 'front', 'wrist_left', 'wrist_right']
    pcd_type = 'rgb_pcd_rps6144'

    if task in ["bimanual_push_box", "bimanual_sweep_to_dustpan"]:
        bounding_box = np.array([[-0.5, -0.55, 0.75], [1.1, 0.55, 1.98]])
    else:
        bounding_box = np.array([[-0.5, -0.55, 0.77], [1.1, 0.55, 1.98]])

    print(cameras)
    print(pcd_type)
    print(bounding_box)

    MAX_THREADS = 10
    chunk_size = 10
    
    worker_threads = []
    for i in range(MAX_THREADS):
        start = i * chunk_size
        end = min(start + chunk_size - 1, 99)
        thread = threading.Thread(target=getpcd.process_episodes, args=(start, end, cameras, pcd_type, bounding_box))
        worker_threads.append(thread)
        thread.start()

    for thread in worker_threads:
        thread.join()
