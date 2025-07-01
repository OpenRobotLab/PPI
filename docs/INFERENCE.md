# Inference

## Inference Guide
### Step 1: Download the test dataset
For the test data, we use the official RLBench2 dataset, which can be found [here](https://bimanual.github.io/).

### Step 2: Install xvfb(if running on server without display)
```bash
sudo apt install xvfb
sudo apt install qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools
```

### Step 3: Modify Configuration Files

#### 1. Update `PPI/inference-for-rlbench2/conf/method/PPI.yaml`
```yaml
sam_checkpoint_path: the path to the SAM checkpoint
gdino_config_path: the path to the GroundingDINO config file in repo `GroundingDINO`
gdino_checkpoint_path: the path to the GroundingDINO checkpoint
instruction_embeddings_path: the path to the instruction embeddings
```

#### 2. Update `PPI/inference-for-rlbench2/conf/eval_ppi.yaml`
```yaml
framework.logdir: the path to the log directory(for saving the videos)
framework.weightsdir: the path to the weights directory
```

#### 3. Modify the following variables in inference scripts in `scripts/inference/`:
```yaml
framework.weight_name: the name of the weight
framework.ckpt_name: the name of the ckpt
rlbench.demo_path: the path to the test dataset
cinematic_recorder.save_path: the path to save the videos
```

#### 4. Modify `YOUR_PATH_TO_COPPELIASIM` in the line 37 in `PPI/inference-for-rlbench2/eval_ppi.py`:
```python
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = YOUR_PATH_TO_COPPELIASIM 
```


### Step 4: Run the inference script

First, start a virtual X server (Xvfb) on display :99 with a screen resolution of 1024x768 and 16-bit color depth, running in background.

```bash
Xvfb :99 -screen 0 1024x768x16 &
```

Then, run the inference script. The inference script is located in the `scripts/inference` directory. For example, to evaluate the ball task, run the following command:

```bash
bash scripts/inference/evaluate_ppi_ball.sh
```

## Download checkpoints
Our checkpoints is available [here](https://huggingface.co/datasets/yuyinyang3y/Open-PPI/tree/main/ckpt).