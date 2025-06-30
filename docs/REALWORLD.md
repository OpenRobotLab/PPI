# Real-World Training and Deployment

## Collect Your Real Datas
**Note**: Code segments containing `TODO` comments may require manual configuration.

### Step 1: Perform Camera Calibration

We use three **Realsense D435i** cameras (left, right, head) to capture RGB-D images. You need to first define the world coordinate and calibrate the extrinsic parameters of three cameras. Format as shown in [view1_left_calibration](real_world_deployment/calibration_results/view1_left_calibration.json).

### Step 2: Prepare Your Tele-Operation System

You need  to prepare your own tele-operation system and modify the **TeleController** in [tele_controller.py](real_world_deployment/utils/controllers/tele_controller.py).

### Step 3: Start To Collect

We use [Frankx](https://github.com/pantor/frankx) as our high-level motion library for collaborative robots.

```bash
python real_world_deployment/tele_control_loop.py
```
When collecting, you can press the following keys:

```bash
S : Start recording.
K : Add keyframe.
C : Stop recording.
Enter : Save data.
Q : Discard data.
```

### Step 4: Get Language Embeddings

Add the instructions of your task in [get_lang_embedding.py](real_world_training_utils/get_lang_embedding.py) and run it:

```bash
python real_world_training_utils/get_lang_embedding.py
```

Then you will get language embeddings `pretrained_models/instruction_embeddings_real.pkl`.

### Step 5: Extract Object Motion from Data

We use [FoundationPose](https://github.com/NVlabs/FoundationPose) and [BundleSDF](https://github.com/NVlabs/BundleSDF) to get the 6d pose of objects in real world. Then the groundtruth of pointflow can be easy to obtained. Just follow this [repo](https://github.com/HealorCai/live-pose) to get the 6d pose information.


After completing the above steps, you will get the data in structure like:
```bash
real_data/                                    
└── training_raw
    └── handover_and_insert_the_plate
        ├── episode0000
        │   └── steps
        │       ├── 0000
        │       │   ├── head_depth_x10000_uint16.png
        │       │   ├── head_rgb.jpg
        │       │   ├── left_depth_x10000_uint16.png
        │       │   ├── left_rgb.jpg
        │       │   ├── other_data.pkl
        │       │   ├── right_depth_x10000_uint16.png
        │       │   └── right_rgb.jpg
        │       ├── 0001
        │       └── 0002
        ├── episode0001
        ├── episode0002
        └── obj_6dpose
            ├── episode0000
            │   └── obj_6dpose.pkl
            ├── episode0001
            └── episode0002
```

## Generate Necessary Labels

**Note**: Code segments containing `TODO` comments may require manual configuration.

### Step 1: Point Cloud

Specify your target task name and replace the dataset source path and output path for generated point clouds with your own paths.

```bash
python scripts/data_generation_real/save_ptc.py
```
### Step 2: Dino Feature

Specify your target task name. If you didn't use the default path, please change the model path, repo path, dataset source path, the point cloud path and the output path for generated dino features with your own paths.

```bash
python scripts/data_generation_real/save_dino.py
```

### Step 3: Point Flow

Specify your target task name. If you didn't use the default path, please change the model path, repo path, dataset source path, the point cloud path, dino feature path and the output path for generated point flow with your own paths.

```bash
python scripts/data_generation_real/save_point_flow_single_obj.py
```

If your task include muti-objects, you can use the muti-objs version.

```bash
python scripts/data_generation_real/save_point_flow_multi_obj.py
```
### Step 4: Norm Stats
Specify your target task name. If you didn't use the default path, please change the point cloud path, dino feature path, the point flow path and the output path for generated norm stats with your own paths.
```bash
python scripts/data_generation_real/save_norm_stats.py
```

## Training

**Note**: Code segments containing `TODO` comments may require manual configuration.

Please check training scripts in `scripts/training_real` and modify the path and wandb keys.

```bash
bash scripts/training_real/handover_and_insert_the_plate.sh
```

Checkpoints are saved to `exp_logs_real/ckpt` by default. To customize the path, modify `ppi/config/ppi_real.yaml`.

## Deploy in Real World
**Note**: Code segments containing `TODO` comments may require manual configuration.

### Step 1: Transfer your Checkpoints

Copy the config file and chechpoints from `exp_logs_real/ckpt` to `real_world_deployment/ckpts` like:

```bash
ckpts
└── PPI
    └── handover_and_insert...ple_512_64_h3_s10_seed0
        ├── config.yaml
        └── epoch4800_model.pth.tar
```

### Step 2: Install PPI

```bash
cd real_world_deployment/policy/PPI
pip install -e .
```

### Step 3: Test in Real World

Specify your task instruction, config_path, ckpt_path, text prompt of object and the path to workspace. 

```bash
python real_world_deployment/inference_control_loop.py
```