if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
from omegaconf import OmegaConf, DictConfig
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import yaml
import tqdm
import numpy as np
from termcolor import cprint
import time
from ppi.policy.ppi_real import PPI
from ppi.dataset.base_dataset import BaseDataset
from ppi.common.checkpoint_util import TopKCheckpointManager
from ppi.common.pytorch_util import dict_apply, optimizer_to
from ppi.model.diffusion.ema_model import EMAModel
from ppi.model.common.lr_scheduler import get_scheduler
from ppi.utils.distributed import init_distributed_device, world_info_from_env
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pdb import set_trace
OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainPPIWorkspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None
        # set torchrun variables
        self.local_rank = int(os.environ["LOCAL_RANK"]) 
        self.global_rank = int(os.environ["RANK"])
        self.distributed = True
        self.sync_bn = True
        self.Cuda = True
        self.find_unused_parameters = True
        self.ngpus_per_node = torch.cuda.device_count()
        
        # set seed
        seed = cfg.training.seed + self.local_rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)

        # configure model
        self.model: PPI = hydra.utils.instantiate(cfg.policy)

        self.ema_model: PPI = None 
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except: 
                self.ema_model = hydra.utils.instantiate(cfg.policy)


        self.optimizer = None
        self.lr_scheduler = None

        # configure training state
        self.global_step = 0
        self.epoch = 0


    def run(self):
        cfg = copy.deepcopy(self.cfg)
        device = torch.device("cuda", self.local_rank)


        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10 
            cfg.training.max_val_steps = 3 
            cfg.training.rollout_every = 20 
            cfg.training.checkpoint_every = 1 
            cfg.training.val_every = 1 
            cfg.training.sample_every = 1 
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True 
        else:
            RUN_ROLLOUT = False 
            RUN_CKPT = True
            verbose = False
        
        RUN_VALIDATION = False # reduce time cost
        
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if os.path.exists(lastest_ckpt_path):
                if self.local_rank == 0:
                    print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.checkpoint_dict_resume = self.load_checkpoint(path=lastest_ckpt_path)



        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)

        assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
        
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True) 
        cfg.dataloader.batch_size = cfg.dataloader.batch_size // self.ngpus_per_node
        cfg.dataloader.shuffle = False 
        train_dataloader = DataLoader(dataset, **cfg.dataloader, sampler=train_sampler) 

        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        cfg.val_dataloader.batch_size = cfg.val_dataloader.batch_size // self.ngpus_per_node
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader, sampler=val_sampler)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        if cfg.training.resume:
            self.load_checkpoint_optimizer(self.checkpoint_dict_resume)

        if self.local_rank == 0:
            print("train_dataloader shape: ", len(train_dataloader) )

        # configure lr scheduler
        self.lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step-1
        )

        if cfg.training.resume:
            self.load_checkpoint_lr_scheduler(self.checkpoint_dict_resume)


        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:

            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure model in ddp
        if self.sync_bn and self.ngpus_per_node > 1 and self.distributed:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        elif self.sync_bn:
            print("Sync_bn is not support in one gpu or not distributed.")

        if self.Cuda:
            if self.distributed:
                self.model = self.model.cuda(self.local_rank)
                self.model = DDP(self.model,find_unused_parameters=self.find_unused_parameters)
                self.ema_model = self.ema_model.cuda(self.local_rank)
            else:
                print("!!! Not distributed !!!")
        
        cfg.logging.name = str(cfg.logging.name)
        if self.local_rank == 0:
            cprint("-----------------------------", "yellow")
            cprint(f"[WandB] group(exp_name): {cfg.logging.group}", "yellow")
            cprint(f"[WandB] name(seed): {cfg.logging.name}", "yellow")
            cprint("-----------------------------", "yellow")
            cprint(f"output_dir: {self.output_dir}", "yellow")


        if self.local_rank == 0:
            with open(os.path.join(self.output_dir, 'config.yaml'), 'w') as f:
                yaml.dump(OmegaConf.to_container(cfg, resolve=True), f)

            # ========================== wandb ============================ 
            # configure logging
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                },
                allow_val_change=True
            )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        # training loop
        for local_epoch_idx in range(cfg.training.num_epochs):
            if self.local_rank == 0:
                step_log = dict()
                train_losses = list()
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                if self.distributed:
                    train_sampler.set_epoch(local_epoch_idx)
                for batch_idx, batch in enumerate(tepoch):
                    t1 = time.time()
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                    if train_sampling_batch is None:
                        train_sampling_batch = batch
                
                    # compute loss
                    t1_1 = time.time()
                    raw_loss, loss_dict = self.model(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()
                    
                    t1_2 = time.time()

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.lr_scheduler.step()
                    t1_3 = time.time()
                    # update ema
                    if cfg.training.use_ema:
                        ema.step(self.model.module)
                    t1_4 = time.time()
                    # logging
                    if self.local_rank==0:
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': self.lr_scheduler.get_last_lr()[0]
                        }
                        t1_5 = time.time()
                        step_log.update(loss_dict)
                    t2 = time.time()
                    
                    if verbose and self.local_rank == 0:
                        print(f"total one step time: {t2-t1:.3f}")
                        print(f" compute loss time: {t1_2-t1_1:.3f}")
                        print(f" step optimizer time: {t1_3-t1_2:.3f}")
                        print(f" update ema time: {t1_4-t1_3:.3f}")
                        print(f" logging time: {t1_5-t1_4:.3f}")

                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        self.global_step += 1

                        if self.local_rank == 0:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            pass

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break

            # at the end of each epoch
            # replace train_loss with epoch average
            if self.local_rank==0:
                train_loss = np.mean(train_losses) 
                step_log['train_loss'] = train_loss

            # ========= eval for this epoch ==========
            # if self.local_rank==0:
            policy = self.model 
            if cfg.training.use_ema:
                policy = self.ema_model
            policy.eval()

            # run validation
            if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                with torch.no_grad():
                    val_losses = list()
                    with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        if self.distributed:
                            val_sampler.set_epoch(local_epoch_idx)
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            loss, loss_dict = self.model(batch)
                            val_losses.append(loss)
                            if (cfg.training.max_val_steps is not None) \
                                and batch_idx >= (cfg.training.max_val_steps-1):
                                break
                    if len(val_losses) > 0 and self.local_rank==0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()
                        step_log['val_loss'] = val_loss

            # run diffusion sampling on a training batch
            if (self.epoch % cfg.training.sample_every) == 0:
                with torch.no_grad():
                    # sample trajectory from training set, and evaluate difference
                    batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                    obs_dict = batch['obs']
                    gt_action = batch['action']
                    if cfg.policy.predict_object_pose and cfg.policy.predict_point_flow:
                        # set_trace()
                        gt_object_pose = batch['object_pose'][:,-cfg.horizon_keyframe:,:]
                        gt_point_flow = batch['point_flow'][:,-cfg.horizon_keyframe:,:,:]
                        b, k, N, _ = gt_point_flow.shape
                        gt_point_flow = gt_point_flow.reshape(b,k*N,3)
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        pred_object_pose = result['object_pose_pred']
                        pred_point_flow = result['point_flow_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        mse_object_pose = torch.nn.functional.mse_loss(pred_object_pose, gt_object_pose)
                        mse_point_flow = torch.nn.functional.mse_loss(pred_point_flow, gt_point_flow)
                        print(f"pred_point_flow{pred_point_flow}")
                        print(f"gt_point_flow{gt_point_flow}")
                        left_pos_l2 = ((pred_action[..., :3] - gt_action[..., :3]) ** 2).sum(-1).sqrt()
                        left_quat_l1 = (pred_action[..., 3:7] - gt_action[..., 3:7]).abs().sum(-1)
                        right_pos_l2 = ((pred_action[..., 7:10] - gt_action[..., 7:10]) ** 2).sum(-1).sqrt()
                        right_quat_l1 = (pred_action[..., 10:14] - gt_action[..., 10:14]).abs().sum(-1)
                        pos_l2_object = ((pred_object_pose[..., :3] - gt_object_pose[..., :3]) ** 2).sum(-1).sqrt()
                        pos_l2_point_flow = ((pred_point_flow[..., :3] - gt_point_flow[..., :3]) ** 2).sum(-1).sqrt()
                        quat_l1_object = (pred_object_pose[..., 3:7] - gt_object_pose[..., 3:7]).abs().sum(-1)
                        if self.local_rank==0:
                            step_log['train_action_mse_error'] = mse.item()
                            step_log['train_object_pose_mse_error'] = mse_object_pose.item()
                            step_log['train_point_flow_mse_error'] = mse_point_flow.item()
                            step_log['left_pos_acc_001'] = (left_pos_l2 < 0.01).float().mean()
                            step_log['left_rot_acc_0025'] = (left_quat_l1 < 0.01).float().mean()
                            step_log['right_pos_acc_001'] = (right_pos_l2 < 0.01).float().mean()
                            step_log['right_rot_acc_0025'] = (right_quat_l1 < 0.01).float().mean()
                            step_log['object_pos_acc_001'] = (pos_l2_object < 0.01).float().mean()
                            step_log['object_rot_acc_0025'] = (quat_l1_object < 0.01).float().mean()
                            step_log['point_flow_pos_acc_0s1'] = (pos_l2_point_flow < 0.01).float().mean()
                            
                    elif cfg.policy.predict_object_pose:
                        # set_trace()
                        gt_object_pose = batch['object_pose'][:,-cfg.horizon_keyframe:,:]
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        pred_object_pose = result['object_pose_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        mse_object_pose = torch.nn.functional.mse_loss(pred_object_pose, gt_object_pose)
                        left_pos_l2 = ((pred_action[..., :3] - gt_action[..., :3]) ** 2).sum(-1).sqrt()
                        left_quat_l1 = (pred_action[..., 3:7] - gt_action[..., 3:7]).abs().sum(-1)
                        right_pos_l2 = ((pred_action[..., 7:10] - gt_action[..., 7:10]) ** 2).sum(-1).sqrt()
                        right_quat_l1 = (pred_action[..., 10:14] - gt_action[..., 10:14]).abs().sum(-1)
                        pos_l2_object = ((pred_object_pose[..., :3] - gt_object_pose[..., :3]) ** 2).sum(-1).sqrt()
                        quat_l1_object = (pred_object_pose[..., 3:7] - gt_object_pose[..., 3:7]).abs().sum(-1)
                        if self.local_rank==0:
                            step_log['train_action_mse_error'] = mse.item()
                            step_log['train_object_pose_mse_error'] = mse_object_pose.item()
                            step_log['left_pos_acc_001'] = (left_pos_l2 < 0.01).float().mean()
                            step_log['left_rot_acc_0025'] = (left_quat_l1 < 0.01).float().mean()
                            step_log['right_pos_acc_001'] = (right_pos_l2 < 0.01).float().mean()
                            step_log['right_rot_acc_0025'] = (right_quat_l1 < 0.01).float().mean()
                            step_log['object_pos_acc_001'] = (pos_l2_object < 0.01).float().mean()
                            step_log['object_rot_acc_0025'] = (quat_l1_object < 0.01).float().mean()

                    elif cfg.policy.predict_point_flow:
                        # set_trace()
                        gt_point_flow = batch['point_flow'][:,-cfg.horizon_keyframe:,:,:]
                        b, k, N, _ = gt_point_flow.shape
                        gt_point_flow = gt_point_flow.reshape(b,k*N,3)
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        pred_point_flow = result['point_flow_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        mse_point_flow = torch.nn.functional.mse_loss(pred_point_flow, gt_point_flow)
                        left_pos_l2 = ((pred_action[..., :3] - gt_action[..., :3]) ** 2).sum(-1).sqrt()
                        left_quat_l1 = (pred_action[..., 3:7] - gt_action[..., 3:7]).abs().sum(-1)
                        right_pos_l2 = ((pred_action[..., 7:10] - gt_action[..., 7:10]) ** 2).sum(-1).sqrt()
                        right_quat_l1 = (pred_action[..., 10:14] - gt_action[..., 10:14]).abs().sum(-1)
                        pos_l2_point_flow = ((pred_point_flow[..., :3] - gt_point_flow[..., :3]) ** 2).sum(-1).sqrt()
                        if self.local_rank==0:
                            step_log['train_action_mse_error'] = mse.item()
                            step_log['train_point_flow_mse_error'] = mse_point_flow.item()
                            step_log['left_pos_acc_001'] = (left_pos_l2 < 0.01).float().mean()
                            step_log['left_rot_acc_0025'] = (left_quat_l1 < 0.01).float().mean()
                            step_log['right_pos_acc_001'] = (right_pos_l2 < 0.01).float().mean()
                            step_log['right_rot_acc_0025'] = (right_quat_l1 < 0.01).float().mean()
                            step_log['point_flow_pos_acc_01'] = (pos_l2_point_flow < 0.01).float().mean()
                    else:
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        left_pos_l2 = ((pred_action[..., :3] - gt_action[..., :3]) ** 2).sum(-1).sqrt()
                        left_quat_l1 = (pred_action[..., 3:7] - gt_action[..., 3:7]).abs().sum(-1)
                        right_pos_l2 = ((pred_action[..., 7:10] - gt_action[..., 7:10]) ** 2).sum(-1).sqrt()
                        right_quat_l1 = (pred_action[..., 10:14] - gt_action[..., 10:14]).abs().sum(-1)
                        if self.local_rank==0:
                            step_log['train_action_mse_error'] = mse.item()
                            step_log['left_pos_acc_001'] = (left_pos_l2 < 0.01).float().mean()
                            step_log['left_rot_acc_0025'] = (left_quat_l1 < 0.01).float().mean()
                            step_log['right_pos_acc_001'] = (right_pos_l2 < 0.01).float().mean()
                            step_log['right_rot_acc_0025'] = (right_quat_l1 < 0.01).float().mean()
                    del batch
                    del obs_dict
                    del gt_action
                    del result
                    del pred_action
                    del mse
                    del left_pos_l2
                    del left_quat_l1
                    del right_pos_l2
                    del right_quat_l1
                    if cfg.policy.predict_object_pose:
                        del pred_object_pose
                        del gt_object_pose
                        del mse_object_pose
                        del pos_l2_object
                        del quat_l1_object

            if ((self.epoch % cfg.training.checkpoint_every) == 0 or self.epoch == 20 or self.epoch == 40) and cfg.checkpoint.save_ckpt and self.local_rank==0:
                # checkpointing
                if cfg.checkpoint.save_last_ckpt:
                    self.save_only_model(tag=f"epoch{self.epoch}")
                if cfg.checkpoint.save_last_snapshot:
                    self.save_snapshot()

                # sanitize metric names
                metric_dict = dict()
                for key, value in step_log.items():
                    new_key = key.replace('/', '_')
                    metric_dict[new_key] = value
                    
            if ((self.epoch % cfg.training.checkpoint_every) == 0 ) and cfg.checkpoint.save_ckpt and self.local_rank==0:        
                self.save_checkpoint()

            # ========= eval end for this epoch ==========
            policy.train()

            if self.local_rank==0:
                for key, value in step_log.items():
                    wandb_run.log(step_log, step=self.global_step)
                    pass
                del step_log

            self.global_step += 1
            self.epoch += 1

            dist.barrier() 
        
    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    

    def save_checkpoint(self, path=None, tag='latest', use_thread=False):
        if path is None:
            path = os.path.join(self.output_dir, 'checkpoints', f'{tag}.pth.tar')

        if not os.path.exists(f"{self.output_dir}/checkpoints"):
            os.makedirs(f"{self.output_dir}/checkpoints")

        model_path =  path.split('.pth.tar')[0] + '_model.pth.tar'
        models_dict = {
            "model_state_dict": self.model.module.state_dict(),
            "ema_model_state_dict": self.ema_model.state_dict()
        }

        checkpoint_dict = {
            "model_state_dict": self.model.module.state_dict(),
            "ema_model_state_dict": self.ema_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "_output_dir": self._output_dir,
            "global_step": self.global_step,
            "epoch": self.epoch
        } 
        torch.save(checkpoint_dict, path)
        torch.save(models_dict, model_path)
        del checkpoint_dict
        del models_dict

        torch.cuda.empty_cache()
        
        return path
    
    def save_only_model(self, path=None, tag='latest', use_thread=False):
        if path is None:
            path = os.path.join(self.output_dir, 'checkpoints', f'{tag}.pth.tar')

        if not os.path.exists(f"{self.output_dir}/checkpoints"):
            os.makedirs(f"{self.output_dir}/checkpoints")

        model_path =  path.split('.pth.tar')[0] + '_model.pth.tar'
        models_dict = {
            "model_state_dict": self.model.module.state_dict(),
            "ema_model_state_dict": self.ema_model.state_dict()
        }
        torch.save(models_dict, model_path)
        del models_dict
        torch.cuda.empty_cache()
        
        return path
    
    def load_checkpoint(self, path=None, tag='latest', **kwargs):
        checkpoint_dict =  torch.load(path)
        self.model.load_state_dict(checkpoint_dict['model_state_dict']) 
        self.ema_model.load_state_dict(checkpoint_dict['ema_model_state_dict'])
        self._output_dir = checkpoint_dict['_output_dir']
        self.global_step = checkpoint_dict['global_step']
        self.epoch = checkpoint_dict['epoch']
        return checkpoint_dict
    
    def load_checkpoint_optimizer(self, checkpoint_dict_resume):
        self.optimizer.load_state_dict(checkpoint_dict_resume['optimizer_state_dict'])
        return checkpoint_dict_resume



    def load_checkpoint_lr_scheduler(self, checkpoint_dict_resume):
        self.lr_scheduler.load_state_dict(checkpoint_dict_resume['lr_scheduler_state_dict'])
        return checkpoint_dict_resume


    def get_checkpoint_path(self, tag='latest'):
        if tag=='latest':
            return os.path.join(self.output_dir, 'checkpoints', f'{tag}.pth.tar')
        elif tag=='best': 
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                score = float(ckpt.split('test_mean_score=')[1].split('.ckpt')[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return os.path.join(self.output_dir, 'checkpoints', best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")

    
def parse_ddp_args():
    args = dict()
    args['local_rank'] = 0
    args['dist_url'] = "env://"
    args['dist_backend'] = "nccl"
    args['no_set_device_rank'] = False
    args['horovod'] = False
    return args


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath('ppi', 'config')),
    config_name='ppi_real'
)
def main(cfg):
    # print(cfg)
    workspace = TrainPPIWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":    
    # distributed training args
    ddp_args = parse_ddp_args()
    # print(args)
    ddp_args['local_rank'], ddp_args['rank'], ddp_args['world_size'] = world_info_from_env()
    device_id = init_distributed_device(ddp_args)
    if int(os.environ["LOCAL_RANK"]) == 0:
        print("Available GPUs:", torch.cuda.device_count())
        tot0 = time.time()

    print("device_id: ", device_id)

    main()

    if int(os.environ["LOCAL_RANK"]) == 0:
        tot1 = time.time()
        print(f"total time: {tot1-tot0:.3f}s")

    dist.destroy_process_group()

