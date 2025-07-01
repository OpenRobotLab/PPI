source ~/.bashrc
conda activate ppi
# Xvfb :99 -screen 0 1024x768x16 &
export DISPLAY=:99
CUDA_VISIBLE_DEVICES=4 python inference-for-rlbench2/eval_ppi.py \
    framework.eval_from_eps_number=0 \
    framework.eval_episodes=100 \
    framework.csv_logging=True \
    framework.tensorboard_logging=false \
    framework.eval_type=1101 \
    framework.weight_name="your/weight/name" \
    framework.ckpt_name="your/ckpt/name" \
    framework.jump_step=1 \
    rlbench.headless=False \
    rlbench.episode_length=400 \
    rlbench.task_name="dustpan" \
    rlbench.tasks=[bimanual_sweep_to_dustpan]  \
    rlbench.demo_path='your/test/demo/dir' \
    rlbench.include_lang_goal_in_obs=true \
    rlbench.query_freq=25 \
    method.policy.horizon_keyframe=4    \
    method.policy.horizon_continuous=50  \
    method.policy.n_obs_steps=1 \
    method.policy.n_action_steps=54 \
    method.policy.bounding_box=[[-0.5,-0.55,0.75],[1.1,0.55,1.98]] \
    method.policy.fps_num=6144 \
    method.policy.prediction_type='keyframe_continuous' \
    method.policy.what_condition='ppi' \
    method.policy.pointflow_num=200  \
    method.policy.text_prompt="a top-down view of a brown T-shaped pole and its bottom."    \
    method.policy.prompt_type="box"    \
    method.policy.sample_type="rps" \
    method.policy.num_inference_steps=1000 \
    method.policy.sam_cameras=["over_shoulder_left"]    \
    cinematic_recorder.save_path="your/video/log/dir"
