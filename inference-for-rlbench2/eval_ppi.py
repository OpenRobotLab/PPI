import gc
import logging
import os
import sys

import peract_config

import hydra
import numpy as np
import torch
import pandas as pd
from omegaconf import DictConfig, OmegaConf, ListConfig
from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
from rlbench.action_modes.action_mode import BimanualJointPositionActionMode
from rlbench.action_modes.arm_action_modes import BimanualEndEffectorPoseViaPlanning
from rlbench.action_modes.arm_action_modes import BimanualJointPosition, JointPosition
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete, BimanualGripperJointPosition
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete

from rlbench.backend import task as rlbench_task
from rlbench.backend.utils import task_file_to_task_class
from yarr.runners.independent_env_runner import IndependentEnvRunner
from yarr.utils.stat_accumulator import SimpleAccumulator

from helpers import utils
from helpers import observation_utils

from yarr.utils.rollout_generator import RolloutGenerator
import torch.multiprocessing as mp

from agents import agent_factory
import yaml
import os
import cv2
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = YOUR_PATH_TO_COPPELIASIM 



def eval_seed(
    eval_cfg, weightsdir, logdir, env_device, multi_task, seed, env_config
) -> None:
    tasks = eval_cfg.rlbench.tasks
    rg = RolloutGenerator()

    agent = agent_factory.create_agent(eval_cfg) # TODO
    stat_accum = SimpleAccumulator(eval_video_fps=30)

    cwd = os.getcwd()

    env_runner = IndependentEnvRunner(
        train_env=None,
        agent=agent, 
        train_replay_buffer=None,
        num_train_envs=0,
        num_eval_envs=eval_cfg.framework.eval_envs,
        rollout_episodes=99999,
        eval_episodes=eval_cfg.framework.eval_episodes,
        training_iterations=eval_cfg.framework.training_iterations,
        eval_from_eps_number=eval_cfg.framework.eval_from_eps_number,
        episode_length=eval_cfg.rlbench.episode_length,
        stat_accumulator=stat_accum,
        weightsdir=weightsdir,
        logdir=logdir,
        env_device=env_device,
        rollout_generator=rg,
        num_eval_runs=len(tasks), # 1
        multi_task=multi_task, # False
        time_steps=eval_cfg.method.policy.n_obs_steps,
        num_processes=eval_cfg.framework.eval_processes
    )

    env_runner._on_thread_start = peract_config.config_logging

    manager = mp.Manager()
    save_load_lock = manager.Lock()
    writer_lock = manager.Lock()

    if type(eval_cfg.framework.eval_type) == int:
        weight_folders = [int(eval_cfg.framework.eval_type)]
        print("Weight:", weight_folders)

    else:
        raise Exception("Unknown eval type")

    if len(weight_folders) == 0:
        logging.info(
            "No weights to evaluate. Results are already available in eval_data.csv"
        )
        sys.exit(0)

    # evaluate several checkpoints in parallel
    # NOTE: in multi-task settings, each task is evaluated serially, which makes everything slow!
    split_n = utils.split_list(weight_folders, eval_cfg.framework.eval_envs)
    num_processes = eval_cfg.framework.eval_processes
    for split in split_n:
        for e_idx, weight in enumerate(split):
            env_runner.start(
                weight,
                save_load_lock, 
                writer_lock,
                env_config,
                e_idx % torch.cuda.device_count(),
                eval_cfg.framework.eval_save_metrics,
                eval_cfg.cinematic_recorder,0)

    del env_runner
    del agent
    gc.collect()
    torch.cuda.empty_cache()


@hydra.main(config_name="eval_ppi", config_path="conf")
def main(eval_cfg: DictConfig) -> None:
    logging.info("\n" + OmegaConf.to_yaml(eval_cfg))

    start_seed = eval_cfg.framework.start_seed
    logdir = os.path.join(
        eval_cfg.framework.logdir,
        eval_cfg.rlbench.task_name,
        eval_cfg.method.name,
        "seed%d" % start_seed,
    )
    weightsdir = eval_cfg.framework.weightsdir

    env_device = utils.get_device(eval_cfg.framework.gpu)
    logging.info("Using env device %s." % str(env_device))

    gripper_mode = eval(eval_cfg.rlbench.gripper_mode)()
    arm_action_mode = eval(eval_cfg.rlbench.arm_action_mode)()
    action_mode = eval(eval_cfg.rlbench.action_mode)(arm_action_mode, gripper_mode)

    
    is_bimanual = eval_cfg.method.robot_name == "bimanual"

    if is_bimanual:
        # TODO: automate instantiation with eval
        task_path = rlbench_task.BIMANUAL_TASKS_PATH
    else:
        task_path = rlbench_task.TASKS_PATH

    task_files = [
        t.replace(".py", "")
        for t in os.listdir(task_path)
        if t != "__init__.py" and t.endswith(".py")
    ]
    eval_cfg.rlbench.cameras = (
        eval_cfg.rlbench.cameras
        if isinstance(eval_cfg.rlbench.cameras, ListConfig)
        else [eval_cfg.rlbench.cameras]
    )
    obs_config = observation_utils.create_obs_config(
        eval_cfg.rlbench.cameras,
        eval_cfg.rlbench.camera_resolution,
        eval_cfg.method.name,
        eval_cfg.method.robot_name
    )

    if eval_cfg.cinematic_recorder.enabled:
        obs_config.record_gripper_closing = True

    multi_task = len(eval_cfg.rlbench.tasks) > 1

    tasks = eval_cfg.rlbench.tasks
    task_classes = []
    for task in tasks:
        if task not in task_files:
            raise ValueError('Task %s not recognised!.' % task)
        task_classes.append(task_file_to_task_class(task, is_bimanual))
        # eg. CoordinatedLiftBall


    # single-task or multi-task
    if multi_task:
        env_config = (
            task_classes,
            obs_config,
            action_mode,
            eval_cfg.rlbench.demo_path,
            eval_cfg.rlbench.episode_length,
            eval_cfg.rlbench.headless,
            eval_cfg.framework.eval_episodes,
            eval_cfg.rlbench.include_lang_goal_in_obs,
            eval_cfg.rlbench.time_in_state,
            eval_cfg.framework.record_every_n,
        )
    else:
        env_config = (
            task_classes[0],
            obs_config,
            action_mode,
            eval_cfg.rlbench.demo_path,
            eval_cfg.rlbench.episode_length,
            eval_cfg.rlbench.headless,
            eval_cfg.rlbench.include_lang_goal_in_obs,
            eval_cfg.rlbench.time_in_state,
            eval_cfg.framework.record_every_n,
            eval_cfg.rlbench.lang_path,
        )

    logging.info("Evaluating seed %d." % start_seed)
    eval_seed(
        eval_cfg,
        weightsdir,
        logdir,
        env_device,
        multi_task,
        start_seed,
        env_config,
    )


if __name__ == "__main__":
    peract_config.on_init()
    main()

