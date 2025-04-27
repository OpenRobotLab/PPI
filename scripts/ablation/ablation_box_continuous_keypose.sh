source ~/.bashrc
conda activate ppi
export PYTHONUNBUFFERED=1
ngpus=8
wandb_mode=online
export WANDB__SERVICE_WAIT=600
export WANDB_API_KEY=$YOUR_WANDB_API_KEY
export HYDRA_FULL_ERROR=1 
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_LAUNCH_BLOCKING=1

torchrun --nnodes 1 --nproc_per_node $ngpus --master_port 10004 ddp_train.py \
    task='box' \
    name='ablation_ppi_8gpus' \
    addition_info="0417_ablation_keypose_continuous" \
    wandb_name="ppi_box" \
    n_obs_steps=1 \
    n_action_steps=54 \
    policy.use_lang=true \
    policy.what_condition='keypose_continuous' \
    policy.predict_point_flow=false \
    task.dataset.pcd_fps=6144 \
    task.dataset.pcd_type='rgb_pcd_rps6144' \
    task.dataset.point_flow_type='world_ordered_rps200'   \
    task.dataset.kp_num=10 \
    dataloader.batch_size=128 \
    val_dataloader.batch_size=128 \
    training.num_epochs=500 \
    task.dataset.prediction_type='keyframe_continuous' \
    horizon_keyframe=4  \
    horizon_continuous=50   \
    task.dataset.stats_filepath='data/training_processed/norm_stats/norm_stats_bimanual_push_box_rgb_pcd_rps6144_keyframe_continuous_world_ordered_rps200.pth' \

