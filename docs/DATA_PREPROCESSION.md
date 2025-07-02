# Data Preprocession

## Download our regenerated dataset 
Our regenerated dataset can be download from [here](https://huggingface.co/datasets/yuyinyang3y/Open-PPI). We recommand put the training data at `$[YOUR_PATH_TO_PPI]/data/training_raw`.

If you want to regenerate the dataset for other tasks in RLBench2, please refer to [DATA_GENERATION.md](DATA_GENERATION.md).

## Generate necessary labels
Before running code, follow our [Installation Guide](INSTALLATION.md) for dependencies and environment configuration. 

```bash
cd ${YOUR_PATH_TO_PPI}
conda activate ppi
```

**Note**: Code segments containing `TODO` comments may require manual configuration (if you don't use our default path). Below are detailed instructions for these modifications:

### Step 1: Point Cloud

Specify your target task name and replace the dataset source path and output path for generated point clouds with your own paths.

```bash
python scripts/data_generation/save_ptc.py
```
### Step 2: Dino Feature
(1) Download the ckpt for dinov2 model from [here](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth). We recommand put the pretrained model weight at `$[YOUR_PATH_TO_PPI]/pretrained_models/hub/checkpoints/dinov2_vits14_pretrain.pth`.

```bash
mkdir repos
cd repos
git clone https://github.com/facebookresearch/dinov2.git
cd ..
```

(2) Specify your target task name. If you didn't use the default path, please change the model path, repo path, dataset source path, the point cloud path and the output path for generated dino features with your own paths.

```bash
python scripts/data_generation/save_dino.py
```

### Step 3: Point Flow
(1) Download the ckpt for SAM from [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth). We recommand put the pretrained model weight at `$[YOUR_PATH_TO_PPI]/pretrained_models/sam_vit_b_01ec64.pth`.

(2) Download the ckpt for GroundingDino from [here](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth). We recommand put the pretrained model weight at `$[YOUR_PATH_TO_PPI]/pretrained_models/groundingdino_swinb_cogcoor.pth`.

(3) If you haven't installed `segment_anything` and `groundingdino`:
```bash
# install sam
cd repos
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install .
cd ..

# install groundingdino
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install .
cd ..

cd ..
```

(4) Specify your target task name. If you didn't use the default path, please change the model path, repo path, dataset source path, the point cloud path, dino feature path and the output path for generated point flow with your own paths.

(5) **Important:** Uncomment the text prompt and camera list for your selected task.

```bash
python scripts/data_generation/save_point_flow.py
```
### Step 4: Norm Stats
Specify your target task name. If you didn't use the default path, please change the point cloud path, dino feature path, the point flow path and the output path for generated norm stats with your own paths.
```bash
python scripts/data_generation/save_norm_stats.py
```

## Default Structure

```bash
data/                                        # Primary data directory
│
├── training_raw/                            # Raw training datasets
│   └── bimanual_lift_ball/  
│
└── training_processed/                      # Processed training data outputs
    ├── point_cloud/                         # Generated point clouds
    │   └── bimanual_lift_ball/
    │
    ├── dino_feature/                        # Extracted DINO features
    │   └── bimanual_lift_ball/
    │
    ├── point_flow/                          # Computed point flow data
    │   └── bimanual_lift_ball/
    │
    └── norm_stats/                          # Normalization statistics
    │
    └── instruction_embeddings.pkl           # language instruction embeddings
```
