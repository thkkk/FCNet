import sys
sys.path.append('..')

from typing import Tuple, List, Optional, Union
import pprint
import time
import os
from os.path import join, exists
import re
from collections import deque
import math
from tqdm import tqdm
import statistics
from datetime import datetime
import pytz
import collections
import statistics
import socket
import json
import wandb
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset

import numpy as np
np.set_printoptions(suppress=True, formatter={'float_kind': '{:16.3f}'.format}, linewidth=1000_000_000,
                    threshold=np.inf)
# np.set_printoptions(suppress=True)
# np.set_printoptions(suppress=True,formatter={'float_kind':'{:f}'.format})
# np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.5f}'.format})
# np.set_printoptions(threshold = np.inf)

from utils.common import purge, find_file, convert_second_to_format, loss_calc, \
    TimeCounter, sync_mgpu, convert_tensor_to_shape
from utils.type_utils import EPISODE_LOAD_MODES, CHUNK_LOAD_MODES
from models import DecisionTransformer, FourierController
from data.load_data import EpisodeLoader



class Runner:
    retain_graph = True # only valid when in episode train mode
    detect_anomaly = False # only valid when in episode train mode

    def __init__(self, model: Union[FourierController, DecisionTransformer], 
                 scheduler, optimizer, 
                 train_loader: Union[DataLoader, EpisodeLoader], 
                 test_loader: Union[DataLoader, EpisodeLoader],
                 local_rank: int, args: dict = None, device = None, model_config: dict = None):
        # 函数传递参数
        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.local_rank = local_rank
        self.args = args # hydra 传递参数
        self.device = device
        self.model_config = model_config

        # 外部传递参数
        self.epochs = args['epochs']
        
        self.clip = args['clip']
        
        self.tasks = deepcopy(args['tasks'])
        self.tasks.sort()
        self.test_interval = args['test_interval']
        
        self.save_interval = args['save_interval']
        self.save_test_best = args['save_test_best']
        self.use_tensorboard = args['use_tensorboard']
        self.use_wandb = args['use_wandb']
        self.distributed = args['distributed']
        self.data_mode = args['data_mode']

        self.use_fp16 = args['use_fp16']
        self.use_flash_attn = args['use_flash_attn']
                
        # for fourier 
        self.ctx_dim = args['ctx_dim']

        self.use_tensorboard = self.use_tensorboard and (self.local_rank == 0)
        self.use_wandb = self.use_wandb and (self.local_rank == 0)
        
        self.chk_tag = '-'.join(self.tasks)
        self.save_best_last_epoch = None
        
        self.scaler = GradScaler()

        # self.chk_saved_time = '_'.join(str(datetime.now().replace(microsecond=0)).split(' '))
        self.chk_saved_time = datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d_%H-%M-%S')
        self.run_name = self.chk_saved_time if args['time_tag'] is None else args['time_tag']
        self.chk_dir = join(
            args['train_log_dir'],
            self.chk_tag,
            self.run_name,
        )
        self.log_dir = self.chk_dir
        return

    def tensorboard_init(self):
        if self.use_tensorboard:
            # raise NotImplementedError('Tensorboard logger is not Implemented, please use wandb.')
            if self.local_rank == 0:
                os.makedirs(self.log_dir, exist_ok=True)
            self.tensor_board_writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

    def tensor_board_log(self, category: str, name: str, value: Union[float, int], idx: int):
        if self.use_tensorboard:
            self.tensor_board_writer.add_scalar(f"{category}/{name}", value, idx)

    def wandb_init(self):
        if self.use_wandb:
            wandb.login(key='4a2d928dff304204d54de2d4043d347b46e24f2c')
            configs = {}
            for k in self.args:
                configs[k] = self.args[k]
            configs['hostname'] = socket.gethostname()
            if configs['hostname'] in self.args['host_name_mapping']:
                configs['hostname'] = self.args['host_name_mapping'][configs['hostname']]
            configs['run_name'] = self.run_name
            wandb_run = wandb.init(
                project="DT",
                config=configs,)
            
            wandb.define_metric("Train_info/train_step")
            wandb.define_metric("Train_info/epoch")
            wandb.define_metric("Train_info/loss", step_metric='Train_info/train_step')
            wandb.define_metric("Train_info/learning_rate", step_metric='Train_info/train_step')
            wandb.define_metric("Train_info/save_test_best", step_metric='Train_info/epoch')
            wandb.define_metric("Metrics/test_loss_mean", step_metric='Train_info/epoch')

    def wandb_define_metric(self, x, y=None):
        if self.use_wandb:
            if y is None:
                wandb.define_metric(x)
            else:
                wandb.define_metric(x, step_metric=y)

    def wandb_log(self, x, y):
        if self.use_wandb:
            wandb.log({x: y})

    def wandb_finish(self):
        if self.use_wandb:
            wandb.finish()

    def _init_pastkv(self, batch_size: int):
        if self.model_config['model_name'] == 'transformer':
            self.pastkv = []
            n_head, seq_len, hidden_size = self.model_config['n_head'], self.model_config['seq_len'], \
                self.model_config['hidden_size']
            for _ in range(self.model_config['n_layer']):
                self.pastkv.append([])
                for _ in range(2):
                    self.pastkv[-1].append(
                        torch.zeros((batch_size, n_head, seq_len - 1, hidden_size // n_head), 
                                    dtype=torch.float32, device=self.device)
                    )
                self.pastkv[-1] = tuple(self.pastkv[-1])
            self.pastkv = tuple(self.pastkv)
        else:
            raise NotImplementedError
    
    def _init_las_actions(self, batch_size: int):
        self.las_actions = torch.zeros((batch_size, 1, self.model_config['tgt_dim']), dtype=torch.float32,
                                       device=self.device)

    def _reset_pastkv(self, mask: torch.Tensor):
        '''
        @param mask: (num_envs,)
        '''
        mask_ids = torch.nonzero(mask).reshape(-1)
        for i in range(len(self.pastkv)):
            for j in range(len(self.pastkv[0])):
                self.pastkv[i][j][mask_ids, ...] = 0

    def _reset_las_actions(self, mask: torch.Tensor):
        '''
        @param mask: (num_envs,)
        '''
        mask_ids = torch.nonzero(mask).reshape(-1)
        self.las_actions[mask_ids, ...] = 0

    def chunk_test_a_step(
        self, src: torch.Tensor, tgt: torch.Tensor, test_info: dict, print_tensor: bool = False
    ):
        src, tgt = src.to(self.device), tgt.to(self.device)
        src, src_context, tgt, tgt_context, action = self.preprocess_data(src, tgt, "test")
        
        with torch.inference_mode():
            if src_context != None:  # fourier_controller 
                # for fourier_controller: _phys_state + context -> action
                bz, max_len, _ = src.shape
                device = self.device
                self.model.module.reset_recur(bz, device)
                T = src.shape[1]
                action_preds = torch.zeros_like(tgt) # (bz, T, action_dim)
                for t in range(T):
                    action_preds[:, t:t+1, :] = self.model(src[:, t:t+1, :], src_context[:, t:t+1, :])
                loss_result = loss_calc(action_preds, tgt)
                loss = loss_result['loss']
            else:
                with autocast(enabled=self.use_fp16):
                    action_preds = self.model(src)
                    loss = (action_preds - tgt).pow(2).mean()
        loss = sync_mgpu(loss, self.device, 'avg')
        test_info['test_loss_deque'].append(loss.cpu().item())
        
        if self.local_rank == 0 and print_tensor:
            if self.data_mode == 'nonmdp':
                row_idxs = np.random.randint(tgt.shape[0], size=3)
                tick_idx = np.random.randint(tgt.shape[1])
                print_tensor = torch.cat([tgt[row_idxs, tick_idx, :].cpu(),
                                          action_preds[row_idxs, tick_idx, :].cpu()], dim=0)
            elif self.data_mode == 'mdp':
                row_idxs = np.random.randint(tgt.shape[0], size=3)
                print(f"tgt: {tgt.shape}, action: {action_preds.shape}")
                print_tensor = torch.cat([torch.squeeze(tgt[row_idxs, :, :]).cpu(),
                                          torch.squeeze(action_preds[row_idxs, :, :]).cpu()], dim=0)
            print()
            print(print_tensor.numpy())
            print()


    def episode_test_a_step(
        self, src: torch.Tensor, tgt: torch.Tensor, first_step_mask: torch.Tensor, test_info: dict, 
        print_tensor: bool = False, test_loss_valid: bool = True
    ):
        '''
        @param src: (bz, src_dim)
        @param tgt: (bz, tgt_dim)
        @param first_step_mask: (bz,) indicate which env is at the first step of its episode
        '''
        
        ''' move data to GPU & adjust data shape '''
        src, tgt = src.to(self.device), tgt.to(self.device)
        src, tgt = src.unsqueeze(1), tgt.unsqueeze(1)
        
        ''' empty first step envs' KVCache & las_actions '''
        self._reset_pastkv(first_step_mask)
        self._reset_las_actions(first_step_mask)
        
        local_test_loss_deque = []
        with torch.inference_mode():
            with autocast(enabled=self.use_fp16):
                actions, self.pastkv = self.episode_a_step_forward(src, self.pastkv, self.las_actions)
                self.las_actions = actions.detach().clone()
                
                if test_loss_valid:
                    loss = (actions - tgt).pow(2).mean()
                    local_test_loss_deque.append(loss)
        if test_loss_valid:
            loss = torch.stack(local_test_loss_deque, dim=0).mean(dim=0)
            loss = sync_mgpu(loss, self.device, 'avg')
            test_info['test_loss_deque'].append(loss.cpu().item())
        
        # only nonmdp
        if self.local_rank == 0 and print_tensor:
            def merge_print_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
                return torch.cat([tensor1.unsqueeze(1), tensor2.unsqueeze(1)], dim=1).reshape(-1, tensor1.size(-1))
            row_idxs = np.random.randint(actions.size(0), size=5)
            print()
            print(merge_print_tensors(actions[row_idxs, 0, :], tgt[row_idxs, 0, :])\
                .cpu().numpy())
            print()

    def test(self, train_info):
        ''' reshuffle dataloader(temporarily only for episode mode) '''
        if self.model_config['input_data_mode'] == 'episode':
            self.test_loader.reshuffle_dataset()
            self.test_loader.set_iter_max(sync_mgpu(len(self.test_loader), self.device, 'min'))
        
        t = enumerate(self.test_loader)
        if self.local_rank == 0:
            if self.model_config['input_data_mode'] == 'chunk':
                t = tqdm(t, total=len(self.test_loader), leave=True, ncols=150,
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', mininterval=3)
            elif self.model_config['input_data_mode'] == 'episode':
                t = tqdm(t, total=self.test_loader.real_length, leave=True, ncols=150,
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', mininterval=3)
            else:
                raise NotImplementedError

        test_info = dict(
            test_loss_deque = collections.deque(),
        )
        for i, test_data in t:
            # src: state, tgt: action
            if self.model_config['input_data_mode'] == 'chunk':
                self.chunk_test_a_step(*test_data, test_info, i == len(self.test_loader) - 1)
            elif self.model_config['input_data_mode'] == 'episode':
                self._init_pastkv(self.test_loader.batch_size)
                self._init_las_actions(self.test_loader.batch_size)
                # 判断此次数据是否参与计算test_loss
                test_loss_valid = True
                if self.model_config['episode_first_no_backward'] and \
                        i < self.test_loader.max_episode_length:
                    test_loss_valid = False
                self.episode_test_a_step(
                    *test_data, test_info, print_tensor = (i == self.test_loader.real_length - 1),
                    test_loss_valid=test_loss_valid)
            else:
                raise NotImplementedError
        test_loss_mean = statistics.mean(test_info['test_loss_deque'])
        test_info['test_loss_deque'].clear()

        self.tensor_board_log('Metrics', 'test_loss_mean', test_loss_mean, train_info['epoch'])
        self.wandb_log('Metrics/test_loss_mean', test_loss_mean)

        if self.local_rank == 0:
            print(f'test_loss_mean: {test_loss_mean:.6f}')
        return test_loss_mean

    def chunk_train_a_step(self, src: torch.Tensor, tgt: torch.Tensor, t: tqdm, train_info: dict):
        self.wandb_log('Train_info/train_step', train_info['train_step'])
        src, tgt = src.to(self.device), tgt.to(self.device)
        src, src_context, tgt, tgt_context, action = self.preprocess_data(src, tgt)
        # src: (bz, seq_len, src_dim)
        self.optimizer.zero_grad()
        
        if src_context != None:  # fourier_controller
            # for fourier_controller: _phys_state + context -> action
            # if src.shape[0] != self.model_config['batch_size']:
            #     return
            # src.shape[0] == batch_size
            self.model.module.clear_recur_cache()
            self.model.module.set_parall()
            if self.model_config["is_chunk_wise"] == True:
                T = int(self.model_config["seq_len"] / self.model_config["n_layer"])
                action_preds = torch.zeros_like(tgt) # (bz, seq_len, action_dim)
                prev_x_layers = torch.zeros(
                    self.model_config["n_layer"], 
                    src.shape[0],
                    T,
                    self.model_config["fno_hidden_size"]
                )
                prev_x_ft_layers = torch.zeros(
                    self.model_config["n_layer"], 
                    src.shape[0],
                    self.model_config["n_modes"],
                    self.model_config["fno_hidden_size"]
                )
                for i in range(self.model_config["n_layer"]):
                    action_preds[:, i*T:(i+1)*T, :], prev_x_layers, prev_x_ft_layers = \
                        self.model(src[:, i*T:(i+1)*T, :], src_context[:, i*T:(i+1)*T, :],
                                   prev_x_layers, prev_x_ft_layers)
            else:
                action_preds = self.model(src, src_context)[:, -tgt.shape[1]:, :]
            loss_result = loss_calc(action_preds, tgt)
            loss = loss_result['loss']
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        else:
            with autocast(enabled=self.use_fp16):
                action_preds = self.model(src)
                loss = (action_preds - tgt).pow(2).mean()
            if self.use_fp16:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            if self.use_fp16:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.scheduler.step()
        
        loss = sync_mgpu(loss, self.device, 'avg')
        
        self.tensor_board_log('Train_info', 'loss', loss.item(), train_info['train_step'])
        self.tensor_board_log('Train_info', 'learning_rate', self.optimizer.param_groups[0]["lr"], train_info['train_step'])
        self.wandb_log('Train_info/loss', loss.item())
        self.wandb_log('Train_info/learning_rate', self.optimizer.param_groups[0]["lr"])

        if self.local_rank == 0:
            t.set_description(f'Epoch [{train_info["epoch"]}/{self.epochs}]', refresh=False)
            t.set_postfix(
                los = '{:.6f}'.format(loss.item()),
                lr = '{:.8f}'.format(self.optimizer.param_groups[0]["lr"]),
                refresh = False,
            )
            t.update()
        
        train_info['train_step'] += 1

    def episode_a_step_forward(
        self, src: torch.Tensor, pastkv: Tuple[Tuple[torch.Tensor]], las_actions: torch.Tensor
    ):
        inputs = src
        if self.model_config['add_last_action']:
            inputs = torch.cat([inputs, las_actions.detach()], dim=-1)
        
        actions, pastkv = self.model(inputs, use_cache=True, past_key_values=pastkv)
        if pastkv[0][0].size(2) >= self.model_config['seq_len']:
            # pastkv = tuple([tuple([pastkv[i][j][:, :, 1:, :].detach().clone() for j in range(len(pastkv[i]))]) 
            #                 for i in range(len(pastkv))])
            pastkv = tuple([tuple([pastkv[i][j][:, :, 1:, :].detach() for j in range(len(pastkv[i]))]) 
                            for i in range(len(pastkv))])
        return actions, pastkv

    def episode_train_a_step_forward_and_backward(
        self, src: torch.Tensor, tgt: torch.Tensor, pastkv: Tuple[Tuple[torch.Tensor]],
        loss_backward: bool
    ):
        with autocast(enabled=self.use_fp16):
            actions, pastkv = self.episode_a_step_forward(src, pastkv, self.las_actions)
            self.las_actions = actions.detach().clone()

            if loss_backward:
                loss = (actions - tgt).pow(2).mean()
                loss_to_backward = loss.clone()
            else:
                loss = None
        if loss_backward:
            if self.use_fp16:
                self.scaler.scale(loss_to_backward).backward(retain_graph=self.retain_graph)
                self.scaler.unscale_(self.optimizer)
            else:
                loss_to_backward.backward(retain_graph=self.retain_graph)
        return loss, pastkv

    def episode_train_a_step(
        self, src: torch.Tensor, tgt: torch.Tensor, first_step_mask: torch.Tensor, t: tqdm, 
        train_info: dict, loss_backward: bool
    ):
        '''
        @param src: (bz, src_dim)
        @param tgt: (bz, tgt_dim)
        @param first_step_mask: (bz,) indicate which env is at the first step of its episode
        '''
        
        ''' move data to GPU & adjust data shape '''
        src, tgt = src.to(self.device), tgt.to(self.device)
        src, tgt = src.unsqueeze(1), tgt.unsqueeze(1)
        
        ''' empty first step envs' KVCache & las_actions '''
        self._reset_pastkv(first_step_mask)
        self._reset_las_actions(first_step_mask)
        
        ''' this train step '''
        self.wandb_log('Train_info/train_step', train_info['train_step'])
    
        if loss_backward:
            self.optimizer.zero_grad()
        loss, self.pastkv = self.episode_train_a_step_forward_and_backward(
            src, tgt, self.pastkv, loss_backward=loss_backward)

        if loss_backward:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            if self.use_fp16:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.scheduler.step()
            
            loss = sync_mgpu(loss, self.device, 'avg')
        
            self.tensor_board_log('Train_info', 'loss', loss.cpu().item(), train_info['train_step'])
            self.tensor_board_log('Train_info', 'learning_rate', self.optimizer.param_groups[0]["lr"], train_info['train_step'])
            self.wandb_log('Train_info/loss', loss.cpu().item())
            self.wandb_log('Train_info/learning_rate', self.optimizer.param_groups[0]["lr"])

            if self.local_rank == 0:
                t.set_description(f'Epoch [{train_info["epoch"]}/{self.epochs}]', refresh=False)
                t.set_postfix(
                    los = '{:.6f}'.format(loss.cpu().item()),
                    lr = '{:.8f}'.format(self.optimizer.param_groups[0]["lr"]),
                    refresh=False,
                )
                t.update()

            train_info['train_step'] += 1

    def train(self):
        self.tensorboard_init()
        self.wandb_init()
        self.wandb_log('Train_info/epoch', 0)
        train_info = dict(
            train_step = 0,
            test_loss_min = np.inf,
            epoch_run_times = collections.deque(maxlen=100),
            epoch = 1
        )
        
        if self.model_config['input_data_mode'] == 'episode' and self.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
        
        while train_info['epoch'] <= self.epochs:
            with TimeCounter(train_info, 'this_epoch_run_time'):
                ''' 记录epoch到log中 并且配置tqdm进度条 '''
                self.wandb_log('Train_info/epoch', train_info['epoch'])
                
                ''' reshuffle dataloader(temporarily only for episode mode) '''
                if self.model_config['input_data_mode'] == 'episode':
                    self.train_loader.reshuffle_dataset()
                    self.train_loader.set_iter_max(sync_mgpu(len(self.train_loader), self.device, 'min'))
                
                ''' 制作iterator '''
                t = None
                if self.local_rank == 0:
                    if self.model_config['input_data_mode'] == 'chunk':
                        this_epoch_train_steps = len(self.train_loader)
                    elif self.model_config['input_data_mode'] == 'episode':
                        this_epoch_train_steps = self.train_loader.real_length
                    else:
                        raise NotImplementedError
                    
                    if self.model_config['episode_first_no_backward']:
                        this_epoch_train_steps -= self.train_loader.max_episode_length
                    t = tqdm(range(this_epoch_train_steps), total=this_epoch_train_steps, leave=True, 
                            ncols=150, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', mininterval=3,)

                self.train_mode()
                # train_data: Union[Tuple[src, tgt], Tuple[src, tgt, reset_mask]]
                for i, train_data in enumerate(self.train_loader):
                    if self.model_config['input_data_mode'] == 'chunk':
                        self.chunk_train_a_step(*train_data, t, train_info)
                    elif self.model_config['input_data_mode'] == 'episode':
                        self._init_pastkv(self.train_loader.batch_size)
                        self._init_las_actions(self.train_loader.batch_size)
                        # 判断此次数据是否执行loss.backward(), 跳过一个epoch一开始的max_episode_length步
                        # 因为一开始的数据分布不均匀
                        loss_backward = True
                        if self.model_config['episode_first_no_backward'] and \
                                i < self.train_loader.max_episode_length:
                            loss_backward = False
                        self.episode_train_a_step(*train_data, t, train_info, loss_backward=loss_backward)
                    else:
                        raise NotImplementedError
                if self.local_rank == 0:
                    print("This epoch train end!")
            
            ''' 统计这个epoch的运行时间 预估训练剩余时间 '''
            train_info['epoch_run_times'].append(train_info['this_epoch_run_time'])
            remain_time_seconds = statistics.mean(train_info['epoch_run_times']) * \
                (self.epochs - train_info['epoch'] - 1)
            if self.local_rank == 0:
                print('remain_time: {}'.format(convert_second_to_format(remain_time_seconds)))
            
            ''' 测试 '''
            self.eval_mode()
            if train_info['epoch'] % self.test_interval == 0:
                test_loss_res = self.test(train_info)
                if self.local_rank == 0:
                    print("This epoch test end!")

            ''' 每过固定epoch间隔就保存一次模型 '''
            if self.local_rank == 0 and self.save_interval is not None and \
                    train_info['epoch'] % self.save_interval == 0:
                self.save(train_info['epoch'])
            
            ''' 保存best模型 '''
            if train_info['test_loss_min'] > test_loss_res:
                save_test_best_flag = 1
                train_info['test_loss_min'] = test_loss_res
                if self.local_rank == 0 and self.save_test_best:
                    self.save(train_info['epoch'], isBest=True)
            else:
                save_test_best_flag = 0
            self.tensor_board_log('Train_info', 'save_test_best', save_test_best_flag, train_info['epoch'])
            self.wandb_log('Train_info/save_test_best', save_test_best_flag)
            
            ''' 不同训练进程之间进行同步 '''
            # if self.distributed:
            #     sync_mgpu(1, self.device, 'avg')
            
            train_info['epoch'] += 1
        
        self.wandb_finish()
        return

    def save(self, epoch, example_forward_input=None, isBest=False):
        self.eval_mode()
        model = self.model
        distributed = self.distributed
        local_rank = self.local_rank
        chk_dir = self.chk_dir
        model_config = self.model_config

        chk_name = f"{self.chk_tag}_{epoch}_best.pth" if isBest else f"{self.chk_tag}_{epoch}.pth"
        if local_rank == 0 and not os.path.exists(chk_dir):
            os.makedirs(self.chk_dir)
        chk_path = join(chk_dir, chk_name)
        if os.path.exists(chk_path):
            os.remove(chk_path)
        print("Save policy to {}!".format(chk_path))
        
        config_name = 'config.json'
        config_path = join(chk_dir, config_name)
        if not os.path.exists(config_path):
            config_c = json.dumps(model_config, indent=4)
            with open(config_path, 'w') as f:
                f.write(config_c)
        model_state_dict = model.module.state_dict() if distributed else \
            model.state_dict()
        checkpoint = {'model': model_state_dict}
        # print(f"model_state_dict: ")
        # self.print_state_dict(model_state_dict)
        torch.save(checkpoint, chk_path)

        if isBest and self.save_best_last_epoch is not None:
            las_chk_name = f"{self.chk_tag}_{self.save_best_last_epoch}_best.pth"
            las_chk_path = join(chk_dir, las_chk_name)
            os.remove(las_chk_path)
            print(f"Remove best pth at {las_chk_path}.")
        self.save_best_last_epoch = epoch

        # verify model's state_dict
        # model_tmp = DecisionTransformer(src_dim=model_config['src_dim'], tgt_dim=model_config['tgt_dim'],
        #                                 hidden_size=model_config['hidden_size'], n_layer=model_config['n_layer'],
        #                                 n_head=model_config['n_head'], seq_len=model_config['seq_len'],
        #                                 data_mode=model_config['data_mode'], use_fp16=model_config['use_fp16'],
        #                                 use_flash_attn=model_config['use_flash_attn']).to(self.device)
        # model_tmp.load_state_dict(model_state_dict)
        # model_tmp.eval()
        # with torch.inference_mode():
        #     with autocast(self.use_fp16):
        #         output1 = model(example_forward_input)
        #     output2 = model_tmp(example_forward_input)
        #     error_rate = (output1 - output2).pow(2).mean()
        #     # print(f"error_rate: {error_rate:.3f}")
        #     assert (output1 == output2).all(), \
        #         f"save state dict error: {output1 == output2}, error_rate: {error_rate:.6f}"


        # if self.args['export_model_as_jit']:
        #     print(f"export as jit")
        #     model_tmp = DecisionTransformer(src_dim=model_config['src_dim'], tgt_dim=model_config['tgt_dim'],
        #                                     hidden_size=model_config['hidden_size'], n_layer=model_config['n_layer'],
        #                                     n_head=model_config['n_head'], seq_len=model_config['seq_len'],
        #                                     data_mode=model_config['data_mode'], use_fp16=model_config['use_fp16'],
        #                                     use_flash_attn=model_config['use_flash_attn'],
        #                                     is_causal=model_config['is_causal'])
        #     # model_tmp.load_state_dict(model_state_dict)
        #     model_tmp.eval()
        #     exported_jit_model_path = join(chk_dir, 'exported_jit_model.pt')
        #     example_forward_input = example_forward_input[[0], ...].detach().cpu()
        #     print(f"example_forward_input shape: {example_forward_input.shape}")
        #     jit_trace_module = torch.jit.trace(model_tmp, example_forward_input) # torch.jit.trace fix the batch size, so set to 1
        #     with torch.inference_mode():
        #         # module = torch.jit.trace(model.module if distributed else model, example_forward_input)
        #         output1 = model_tmp(example_forward_input)
        #         output2 = jit_trace_module(example_forward_input)
        #         output3 = jit_trace_module(example_forward_input)
        #         # assert (output1 == output2).all(), \
        #         #     f"trace module error1: {output1 == output2}" # error, seems trace do not support Flash Attn
        #         # assert (output2 == output3).all(), \
        #         #     f"trace module error2: {output2 == output3}"
        #         torch.jit.save(jit_trace_module, exported_jit_model_path)
        #     del model_tmp
        return

    def eval_mode(self):
        self.model.eval()
        # if self.distributed:
        #     self.model.module.eval()
        # else:
        #     self.model.eval()
        return

    def train_mode(self):
        self.model.train()
        # if self.distributed:
        #     self.model.module.train()
        # else:
        #     self.model.train()
        return

    def convert_state_dict(self, x:dict):
        res = {}
        for k in x:
            if isinstance(x[k], torch.Tensor):
                res[k] = (x[k].shape, x[k].dtype, x[k].device)
            else:
                res[k] = self.convert_state_dict(x[k])
        return res

    def print_state_dict(self, x:dict):
        pp = pprint.PrettyPrinter()
        pp.pprint(self.convert_state_dict(x))
        return

    def preprocess_data(self, src, tgt, train_or_test='train'):
        src_context = None
        tgt_context = None
        action = tgt[:, :tgt.shape[1], :]
        batch_size = src.shape[0]
        seq_len = tgt.shape[1]
        if self.model_config['model_name'] in ["fourier_controller"]:
            tgt = tgt[:, :, :]  # action: [batch_size, seq_len, action_dim]

            src_context = src[:, :, -self.ctx_dim:]  # context: [batch_size, seq_len, context_dim]
            src = src[:, :, :-self.ctx_dim]  # state: [batch_size, seq_len, state_dim]
            # 之前用的是:src.shape[1]-1，因为历史遗留问题，现在改成了:src.shape[1]
            # if train_or_test == 'train':
                # domain randomization
                # generate random state and random context
                # state_mean, state_var = torch.mean(src, dim=1), torch.var(src, dim=1)
                # context_mean, context_var = torch.mean(src_context, dim=1), torch.var(src_context, dim=1)
                
                # state_random = torch.randn(batch_size, 3 * seq_len, src.shape[2], device=src.device) \
                #     * torch.sqrt(state_var).unsqueeze(1) + state_mean.unsqueeze(1)
                # context_random = torch.randn(batch_size, 3 * seq_len, src_context.shape[2], device=src_context.device) \
                #     * torch.sqrt(context_var).unsqueeze(1) + context_mean.unsqueeze(1)
                
                # padding 0
                # state_random = torch.zeros(batch_size, 3 * seq_len, src.shape[2], device=src.device)
                # context_random = torch.zeros(batch_size, 3 * seq_len, src_context.shape[2], device=src_context.device)
                # tgt_random = torch.zeros(batch_size, 3 * seq_len, tgt.shape[2], device=tgt.device)

                # src = torch.cat((state_random, src), dim=1)
                # src_context = torch.cat((context_random, src_context), dim=1)
                # tgt = torch.cat((tgt_random, tgt), dim=1)
        return src, src_context, tgt, tgt_context, action
    
    # # save model as torch_script_module
    # def save(self, epoch, isBest=False):
    #     model = self.model
    #     distributed = self.distributed
    #     local_rank = self.local_rank
    #     chk_dir = self.chk_dir
        
    #     chk_name = 'model_best.pt' if isBest else 'model_{}.pt'.format(epoch)
    #     if local_rank == 0 and not os.path.exists(chk_dir):
    #         os.makedirs(self.chk_dir)
    #     chk_path = join(chk_dir, chk_name)
    #     if os.path.exists(chk_path):
    #         os.remove(chk_path)
    #     script_module = torch.jit.script(model.module if distributed else model)
    #     torch.jit.save(script_module, chk_path)
    #     print("Save policy to {}!".format(chk_path))
