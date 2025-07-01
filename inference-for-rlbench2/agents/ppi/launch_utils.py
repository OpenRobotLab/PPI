# Adapted from ARM
# Source: https://github.com/stepjam/ARM
# License: https://github.com/stepjam/ARM/LICENSE


from helpers.preprocess_agent import PreprocessAgent
from ppi.policy.ppi import PPI
from agents.ppi.ppi_agent import PPIAgent
from omegaconf import DictConfig
import hydra

def create_agent(cfg: DictConfig):
    actor_net = PPI(
        noise_scheduler_cfg = cfg.method.policy.noise_scheduler_cfg,
        horizon_keyframe = cfg.method.policy.horizon_keyframe, 
        horizon_continuous = cfg.method.policy.horizon_continuous,
        n_action_steps = cfg.method.policy.n_action_steps, 
        n_obs_steps = cfg.method.policy.n_obs_steps, 
        num_inference_steps = cfg.method.policy.num_inference_steps, 
        encoder_output_dim = cfg.method.policy.encoder_output_dim, 
        use_lang = cfg.method.policy.use_lang,
        pointcloud_encoder_cfg = cfg.method.policy.pointcloud_encoder_cfg, 
        use_decoder = False,
        what_condition = cfg.method.policy.what_condition,
        predict_point_flow = cfg.method.policy.predict_point_flow,
    )

    ppi_agent = PPIAgent(
        actor_network=actor_net,
        cameras=cfg.rlbench.cameras,
        task_name=cfg.rlbench.tasks[0],
        weight_name=cfg.framework.weight_name,
        fps_num=cfg.method.policy.fps_num,
        cameras_pcd=cfg.rlbench.cameras_pcd,
        use_pc_color=cfg.method.policy.use_pc_color,
        use_lang = cfg.method.policy.use_lang,
        bounding_box = cfg.method.policy.bounding_box,
        episode_length = cfg.rlbench.episode_length,
        prediction_type = cfg.method.policy.prediction_type,
        save_path = cfg.cinematic_recorder.save_path,
        query_freq = cfg.rlbench.query_freq,
        horizon_continuous = cfg.method.policy.horizon_continuous,
        horizon_keyframe = cfg.method.policy.horizon_keyframe,
        predict_point_flow = cfg.method.policy.predict_point_flow,
        pointflow_num = cfg.method.policy.pointflow_num,
        text_prompt = cfg.method.policy.text_prompt,
        prompt_type = cfg.method.policy.prompt_type,
        sample_type = cfg.method.policy.sample_type,
        sam_cameras = cfg.method.policy.sam_cameras,
        ckpt_name = cfg.framework.ckpt_name,
        jump_step = cfg.framework.jump_step,
        sam_checkpoint_path=cfg.method.policy.sam_checkpoint_path,
        gdino_config_path=cfg.method.policy.gdino_config_path,
        gdino_checkpoint_path=cfg.method.policy.gdino_checkpoint_path,
        instruction_embeddings_path=cfg.method.policy.instruction_embeddings_path
        )

    return PreprocessAgent(pose_agent=ppi_agent,norm_rgb=False)
