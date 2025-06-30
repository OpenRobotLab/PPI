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

torchrun --nnodes 1 --nproc_per_node $ngpus --master_port 11111 ddp_train_real.py \
    task='handover_and_insert_the_plate_simple' \
    name='train_ppi_8gpus' \
    addition_info="YYMMDD-XXXX" \
    wandb_name="real_handover_and_insert_the_plate" \
    n_obs_steps=1 \
    n_action_steps=54 \
    policy.encoder_output_dim=288 \
    policy.use_lang=true \
    policy.what_condition='ppi_real_simple' \
    policy.predict_point_flow=true \
    policy.pointcloud_encoder_cfg.npoint1=256 \
    policy.pointcloud_encoder_cfg.npoint2=64 \
    policy.num_inference_steps=5 \
    task.dataset.pcd_fps=512 \
    task.dataset.pcd_type='rgb_pcd_rps512' \
    task.dataset.point_flow_type='world_ordered_rps50'   \
    task.dataset.kp_num=10 \
    dataloader.batch_size=1024 \
    dataloader.num_workers=8 \
    training.num_epochs=5000 \
    training.checkpoint_every=100 \
    training.sample_every=100 \
    checkpoint.topk.monitor_key='val_loss' \
    checkpoint.topk.mode='min' \
    horizon_keyframe=4  \
    horizon_continuous=50   \
    policy.use_initial_pointflow=true   \
    task.dataset.end=19 \
    task.dataset.hist_num=3 \
    task.dataset.add_openess_sampling=true \
    task.dataset.sample_num=10 \
    task.dataset.stats_filepath='real_data/training_processed_simple/norm_stats/norm_stats_handover_and_insert_the_plate_rgb_pcd_rps512_keyframe_continuous_world_ordered_rps50.pth' \