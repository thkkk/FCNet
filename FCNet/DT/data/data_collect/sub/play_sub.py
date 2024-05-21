class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

import sys
sys.stdout = Unbuffered(sys.stdout)

from typing import List
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import re
from os.path import join, exists
import shutil
import json
import pickle
import lzma
import statistics
import random

import isaacgym
from isaacgym import gymutil, gymapi
from legged_gym.envs import *
from legged_gym.utils import task_registry

import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from FCNet.DT.utils.common import get_availble_gpus, RocordAdder, \
    create_vedio_from_imgs, merge_tensor_dict_to_single_tensor_fixed, \
    MultiAgentActionBuffer, MultiAgentObsBuffer, get_py_virtual_display, \
    get_split_slice, get_split_sz, parse_dt_mode_to_src_tgt_dim, \
    convert_episodes_to_samples_nonmdp_numba, convert_episodes_to_samples_mdp_numba, \
    get_parameter_number, calc_new_env_rew, convert_tensor_to_shape, \
    save_np_mmap, clear_components
from FCNet.DT.utils.aligo_utils import download_folder_to_local
from FCNet.DT.utils.type_utils import CHUNK_LOAD_MODES, EPISODE_LOAD_MODES
from FCNet.DT.utils.data_utils import shuffle_episodes
from FCNet.DT.data.data_collect.sub.evaluate import evaluate

torch.set_printoptions(sci_mode=False, linewidth=100_000)

from collections import deque
import statistics
import time

from tqdm import tqdm
import pprint
pp = pprint.PrettyPrinter()

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "aliengo_stand", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        #{"name": "--task", "type": str, "default": "anymal_c_flat", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
        
        {"name": "--load_optimizer", "action":"store_true"},
        {"name": "--debug_viz", "action": "store_true"},
        {"name": "--reload_std", "action": "store_true", "default": False, "help": ""},
        
        # dt params
        {"name": "--aligo_auto_complete_chks", "action": "store_true", "default": False},
        {"name": "--dt_policy_name", "type": str, "default": None},

        {"name": "--record", "action": "store_true", "default": False},
        {"name": "--ac_ratio", "type": float, "default": None},
        {"name": "--play_ep_cnt", "type": int, "default": None},
        {"name": "--crf", "type": float, "default": None},
        
        {"name": "--save_data", "type": str, "default": None},
        {"name": "--data_dir", "type": str, "default": None},
        {"name": "--local_rank", "type": int, "default": 0},
        {"name": "--processors", "type": int, "default": None},
        {"name": "--total_collect_samples", "type": int, "default": None},
        {"name": "--skip_episodes", "type": int, "default": None},
        {"name": "--reward_limit", "type": float, "default": None},
        {"name": "--eplen_limit", "type": int, "default": None},
        {"name": "--limit_mode", "type": str, "default": None},
        {"name": "--dt_mode", "type": str, "default": None},
        {"name": "--data_mode", "type": str, "default": None},
        {"name": "--seq_len", "type": int, "default": None},
        {"name": "--model_name", "type": str, "default": None},
        
        {"name": "--max_episode_length", "type": int, "default": None},
        
        {"name": "--print_inference_action_time", "action": "store_true", "default": False},

        {"name": "--device_id", "type": int, "default": None, "help": ""},
        {"name": "--legged_gym_version", "type": str, "default": None, "choices": ["old", "new"]},

        {"name": "--push_robots", "action": "store_true", "default": False},
        {"name": "--add_noise", "action": "store_true", "default": False},
        {"name": "--calc_dt_mlp_loss", "action": "store_true", "default": False},
        {"name": "--simplify_print_info", "action": "store_true", "default": False},
        # {"name": "--export_model_as_jit", "action": "store_true", "default": False},
        # {"name": "--use_fp16", "action": "store_true", "default": False},
        # {"name": "--use_flash_attn", "action": "store_true", "default": False},
        # {"name": "--double_v_dim", "action": "store_true", "default": False},
        # {"name": "--is_causal", "action": "store_true", "default": False},
        {"name": "--kv_cache", "action": "store_true", "default": False},
        {"name": "--dummy", "action": "store_true", "default": False},
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)
    
    ''' 获取和设置可用GPU '''
    assert args.device_id is not None, f"args.device_id: {args.device_id}"
    device_id = args.device_id
    device = f'cuda:{device_id}'
    
    args.compute_device_id = device_id
    args.graphics_device_id = device_id
    args.sim_device_id = device_id
    args.rl_device = device
    args.sim_device = device
    
    print(f'use device id: {device_id}')
    return args

def get_action(policy, states, actions_las_t, dt_mode, action_dim,
               data_mode:str, attn_mask=None, tgt_selector=None):
    ''' @param states: (num_envs, seq_len, state_dim) if data_mode == nonmdp else (num_envs, state_dim)
        @param actions_las_t: (num_envs, seq_len, action_dim) if data_mode == nonmdp else (num_envs, action_dim)
        @param attn_mask: (num_envs, seq_len) if data_mode == nonmdp else None
        @param tgt_selector: (num_envs, 1, tgt_dim) if data_mode == nonmdp else None
        @return actions: (num_envs, action_dim)
    '''
    src_mode, tgt_mode = dt_mode.split('_')
    if data_mode == 'nonmdp':
        if src_mode == 'sa':
            srcs = torch.cat((states, actions_las_t), dim=2)
        elif src_mode == 'as':
            srcs = torch.cat((actions_las_t, states), dim=2)
        elif src_mode == 's':
            srcs = states
    elif data_mode == 'mdp':
        if src_mode == 'sa':
            srcs = torch.unsqueeze(torch.cat((states, actions_las_t), dim=1), dim=2)
        elif src_mode == 'as':
            srcs = torch.unsqueeze(torch.cat((actions_las_t, states), dim=1), dim=2)
        elif src_mode == 's':
            srcs = torch.unsqueeze(states, dim=2)

    tgt_preds = policy(srcs)
    if data_mode == 'nonmdp':
        tgt_res = torch.gather(tgt_preds, 1, tgt_selector).squeeze() # (num_envs, action_dim or state_dim+action_dim)
    elif data_mode == 'mdp':
        tgt_res = torch.squeeze(tgt_preds, dim=2)

    if tgt_mode == 'a':
        action_res = tgt_res
    return action_res

def play(args=get_args()):
    if args.local_rank == 0:
        print('args: ')
        pp.pprint(vars(args))

    if args.legged_gym_version == 'old':
        from legged_gym.utils import task_registry
    elif args.legged_gym_version == 'new':
        from legged_robot_personal.envs import task_registry
    
    ''' 外部传递参数 '''
    task = args.task
    data_dir = args.data_dir
    # d_m = args.d_m
    # n_layer = args.n_layer
    # n_head = args.n_head
    play_ep_cnt = args.play_ep_cnt
    dt_policy_name: str = args.dt_policy_name
    aligo_auto_complete_chks = args.aligo_auto_complete_chks
    crf = args.crf
    calc_dt_mlp_loss = args.calc_dt_mlp_loss
    # export_model_as_jit = args.export_model_as_jit
    # use_fp16 = args.use_fp16
    # double_v_dim = args.double_v_dim
    # use_flash_attn = args.use_flash_attn
    # is_causal = args.is_causal
    dummy = args.dummy
    kv_cache = args.kv_cache
    legged_gym_version = args.legged_gym_version

    save_data = args.save_data
    local_rank = args.local_rank # local_rank of tasks
    device_id = args.device_id # GPU device id in real host machine
    processors = args.processors # gpu cnt
    total_collect_samples = args.total_collect_samples
    reward_limit = args.reward_limit
    eplen_limit = args.eplen_limit
    limit_mode = args.limit_mode
    dt_mode = args.dt_mode
    data_mode = args.data_mode
    seq_len = args.seq_len
    model_name = args.model_name
    
    simplify_print_info = args.simplify_print_info
    max_episode_length = args.max_episode_length
    skip_episodes = args.skip_episodes
    
    ''' play save data 加载参数 '''
    if save_data is not None:
        src_mode, tgt_mode = dt_mode.split('_')
    
    ''' play dt 加载chk '''
    if dt_policy_name is not None:
        local_dt_policy_dir = join('../../log', task, dt_policy_name)
        # 使用aligo数据补全chks
        if not exists(local_dt_policy_dir):
            if aligo_auto_complete_chks:
                from aligo import Aligo
                ali = Aligo(name='mk')
                download_folder_to_local(ali, join('/', 'log', 'FCNet', task, dt_policy_name),
                                         join('../../log', task))
            else:
                raise ValueError(f'local chk {task}, {dt_policy_name} not found')
        # 寻找model和config.json
        chk_name = None
        for file_name in os.listdir(local_dt_policy_dir):
            if re.fullmatch(rf'(.*).pth', file_name):
                chk_name = file_name
        assert chk_name is not None, f"{local_dt_policy_dir} do not have *.pth"
        chk_path = join(local_dt_policy_dir, chk_name)
        config_path = join(local_dt_policy_dir, 'config.json')
        assert exists(config_path), f"{local_dt_policy_dir} do not have config.json"
        # 加载chk
        checkpoint = torch.load(chk_path, map_location='cpu')
        with open(config_path, 'r') as f:
            chk_config = json.loads(f.read())
        # 解析config.json
        dt_mode = chk_config['dt_mode']
        src_mode, tgt_mode = dt_mode.split('_')
        model_name = chk_config['model_name']
        data_mode = chk_config['data_mode']
        seq_len = chk_config['seq_len']
        ctx_dim = chk_config['ctx_dim']
    
        print("chk_config")
        pp.pprint(chk_config)
        dt_mode = chk_config['dt_mode']

    ''' 根据外部传递参数配置环境和训练参数 '''
    env_cfg, train_cfg = task_registry.get_cfgs(name=task)

    ''' 录制配置 '''
    if args.record:
        assert args.ac_ratio is not None
        
        # dt_policy_path_seps = dt_policy_path.split('/')
        # record_name = dt_policy_path_seps[dt_policy_path_seps.index('log')+1]
        record_name = f"{task}_{model_name}_{dt_policy_name}"
        
        recode_adder = RocordAdder(args.ac_ratio)

        img_dir = os.path.abspath(join('./record', record_name))
        shutil.rmtree(img_dir, ignore_errors=True)
        os.makedirs(img_dir, exist_ok=True)

        SCREEN_CAPTURE_RESOLUTION = (1920, 1080)
        virtual_display = get_py_virtual_display(size=SCREEN_CAPTURE_RESOLUTION)
        virtual_display.start()
        print(f'Create virtual display! os.environ["DISPLAY"]: {os.environ["DISPLAY"]}')
    
    ''' 覆盖部分环境参数 '''
    env_cfg.seed = np.random.randint(1e9)
    env_cfg.env.num_envs = args.num_envs
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = args.add_noise
    if legged_gym_version == 'new':
        if hasattr(env_cfg, 'debug'):
            env_cfg.debug.debug_viz = args.debug_viz

    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = args.push_robots

    ''' 制作env, 并且设置摄像头的位置和方向 '''
    # Default camera pos: camera_pos: [10, 0, 6], camera_lookat: [11, 5, 3]
    # camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    # camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    env, env_cfg = task_registry.make_env(name=task, args=args, env_cfg=env_cfg)
    if args.record:
        camera_position, camera_lookat = np.array([0, 0, 2.5], dtype=np.float64), \
            np.array([4, 4, 0], dtype=np.float64)
        # print(f'camera_pos: {camera_position}, camera_lookat: {camera_position + camera_direction}')
        print(f'camera_pos: {camera_position}, camera_lookat: {camera_lookat}')
        env.set_camera(camera_position, camera_lookat)
    if (model_name is not None and model_name == 'mlp') or (save_data is not None) or calc_dt_mlp_loss:
        ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=task, args=args, train_cfg=train_cfg)
        mlp_policy = ppo_runner.get_inference_policy(device=env.device)

    print("model_name", model_name)
    ''' play dt 加载模型 '''
    if model_name == 'transformer':
        if chk_config['export_model_as_jit']:
            print(f"load exported jit model.")
            transformer_policy = torch.jit.load(join(local_dt_policy_dir, 'exported_jit_model.pt'),
                                                map_location='cpu')
        else:
            from FCNet.DT.models import DecisionTransformer
            transformer_policy = DecisionTransformer(chk_config)
            get_parameter_number(transformer_policy)
            transformer_policy.load_state_dict(checkpoint['model'])
        print("transformer created")
        transformer_policy = transformer_policy.to(env.device)
        transformer_policy.eval()
    elif model_name == 'retnet':
        from FCNet.DT.models import DecisionRetNet
        retnet_policy = DecisionRetNet(chk_config)
        get_parameter_number(retnet_policy)
        retnet_policy.load_state_dict(checkpoint['model'])
        print("retnet created")
        retnet_policy = retnet_policy.to(env.device)
        retnet_policy.eval()
    elif model_name == 'fourier_controller':
        from FCNet.DT.models.fourier_controller import FourierController
        fourier_policy = FourierController(chk_config)
        print("fourier_policy created")
        get_parameter_number(fourier_policy)
        fourier_policy.load_state_dict(checkpoint['model'])
        fourier_policy = fourier_policy.to(env.device)
        fourier_policy.eval()
    elif model_name == 'mlp_imitation':
        from FCNet.DT.models.mlp_imitation import MLPImitationModel
        mlp_imitation_policy = MLPImitationModel(chk_config)
        get_parameter_number(mlp_imitation_policy)
        mlp_imitation_policy.load_state_dict(checkpoint['model'])
        mlp_imitation_policy = mlp_imitation_policy.to(env.device)
        mlp_imitation_policy.eval()
    elif model_name == 'rnn_imitation':
        from FCNet.DT.models.rnn_imitation import RNNImitationModel
        rnn_imitation_policy = RNNImitationModel(chk_config)
        get_parameter_number(rnn_imitation_policy)
        rnn_imitation_policy.load_state_dict(checkpoint['model'])
        rnn_imitation_policy = rnn_imitation_policy.to(env.device)
        rnn_imitation_policy.eval()

    ''' 获取任务参数 '''
    num_envs = env_cfg.env.num_envs
    state_dim = env_cfg.env.num_observations
    action_dim = env_cfg.env.num_actions
    if max_episode_length is not None and max_episode_length > 0:
        env.max_episode_length = int(max_episode_length)
    max_episode_length = int(env.max_episode_length)
    
    ''' 重置环境 '''
    obs, _ = env.reset()
    
    ''' 初始化 play 环境 '''
    # obs = env.get_observations()
    if legged_gym_version == 'new':
        state_dim = merge_tensor_dict_to_single_tensor_fixed(obs, env, task).shape[-1]

    print(f'task: {task}, num_envs: {num_envs}, state_dim: {state_dim}, action_dim: {action_dim}, \
max_episode_length: {max_episode_length}, legged_gym_version: {legged_gym_version}')
    if dummy: return

    ''' 初始化保存数据设置 '''
    if save_data == 'chunk':
        this_collect_samples = get_split_sz(total_collect_samples, processors, local_rank)
        src_dim, tgt_dim = parse_dt_mode_to_src_tgt_dim(dt_mode, state_dim, action_dim)
        if data_mode == 'nonmdp':
            src_data = np.zeros((this_collect_samples, seq_len, src_dim), dtype=np.float32)
            tgt_data = np.zeros((this_collect_samples, seq_len, tgt_dim), dtype=np.float32)
        elif data_mode == 'mdp':
            src_data = np.zeros((this_collect_samples, src_dim, 1), dtype=np.float32)
            tgt_data = np.zeros((this_collect_samples, tgt_dim, 1), dtype=np.float32)
        state_buf = torch.zeros((num_envs, max_episode_length * 2, state_dim), dtype=torch.float32)
        action_buf = torch.zeros((num_envs, max_episode_length * 2, action_dim), dtype=torch.float32)
        buf_idx = torch.zeros((num_envs, ), dtype=torch.int64)
    elif save_data == 'episode': # nonmdp
        src_dim, tgt_dim = parse_dt_mode_to_src_tgt_dim(dt_mode, state_dim, action_dim)
        this_collect_samples = get_split_sz(total_collect_samples, processors, local_rank)
        this_collect_max_steps = int(this_collect_samples * (max_episode_length + 5))
        save_data_info = {
            'this_collect_samples': this_collect_samples,
            'this_collect_max_steps': this_collect_max_steps,
            'now_collect_steps': int(0),
            'now_collect_episodes': int(0),
            'state_episode_np': np.zeros((this_collect_max_steps, src_dim), dtype=np.float32),
            'action_episode_np': np.zeros((this_collect_max_steps, tgt_dim), dtype=np.float32),
            'episode_length_np': np.zeros((this_collect_samples,), dtype=np.int32),
        }
        def append_episode(episode_state: np.ndarray, episode_action: np.ndarray, save_data_info: dict):
            assert episode_state.shape[0] == episode_action.shape[0], f"{episode_state.shape[0]}, {episode_action.shape[0]}"
            assert dt_mode == 's_a', dt_mode
            eplen = episode_state.shape[0]
            if save_data_info['now_collect_episodes'] < this_collect_samples:
                save_data_info['state_episode_np'][save_data_info['now_collect_steps']: save_data_info['now_collect_steps'] + eplen, :] = \
                    episode_state[:]
                save_data_info['action_episode_np'][save_data_info['now_collect_steps']: save_data_info['now_collect_steps'] + eplen, :] = \
                    episode_action[:]
                save_data_info['episode_length_np'][save_data_info['now_collect_episodes']] = eplen
                save_data_info['now_collect_episodes'] += 1
                save_data_info['now_collect_steps'] += eplen
                return True
            return False
        state_buf = torch.zeros((num_envs, max_episode_length * 2, state_dim), dtype=torch.float32)
        action_buf = torch.zeros((num_envs, max_episode_length * 2, action_dim), dtype=torch.float32)
        buf_idx = torch.zeros((num_envs, ), dtype=torch.int64)

    rewbuffer = deque()
    lenbuffer = deque()
    cur_reward_sum = torch.zeros(env.num_envs, dtype=torch.float32)
    cur_episode_length = torch.zeros(env.num_envs, dtype=torch.int64)

    if model_name == 'transformer':
        if data_mode == 'nonmdp':
            obs_buf = MultiAgentObsBuffer(
                num_envs=env.num_envs,
                seq_len=seq_len,
                max_episode_length=max_episode_length,
                state_dim=state_dim,
                action_dim=action_dim,
                tgt_dim=chk_config['tgt_dim'],
                device=env.device,
                dtype=torch.float32)
            action_buf = MultiAgentActionBuffer(
                num_envs=env.num_envs,
                seq_len=seq_len,
                max_episode_length=max_episode_length,
                action_dim=action_dim,
                device=env.device,
                dtype=torch.float32)
        else:
            actions = torch.zeros((num_envs, action_dim), dtype=torch.float32,
                device=env.device) # for save data storing last action
    
    ''' 再次重置环境 '''
    obs, _ = env.reset()

    # save data (should be mlp)
    if save_data:
        t = tqdm(range(this_collect_samples), total=this_collect_samples,
            leave=True, ncols=150, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
            mininterval=5)
        skip_episodes_valid_flag = (skip_episodes > 0)
        left_episodes_to_skip = torch.full((num_envs,), skip_episodes, dtype=torch.int64)
        while t.n < t.total:
            ''' generate actions '''
            if legged_gym_version == 'old':
                actions = mlp_policy(obs.detach())
            elif legged_gym_version == 'new':
                actions = mlp_policy(obs)

            ''' save data buffer act '''
            if save_data in ['chunk', 'episode']:
                this_state = None
                if legged_gym_version == 'old':
                    this_state = obs.cpu().detach().unsqueeze(1)
                elif legged_gym_version == 'new':
                    this_state = merge_tensor_dict_to_single_tensor_fixed(obs, env, task).cpu().detach().clone().unsqueeze(1)
                    # `dt_mode` is not necessary in save_data, but it is necessary in play
                state_buf.scatter_(1,
                    torch.repeat_interleave(buf_idx.reshape(-1, 1, 1), state_dim, dim=2),
                    this_state)
                action_buf.scatter_(1,
                    torch.repeat_interleave(buf_idx.reshape(-1, 1, 1), action_dim, dim=2),
                    actions.cpu().detach().unsqueeze(1))
                buf_idx += 1

            ''' env step '''
            if legged_gym_version == 'old':
                obs, _, rews, dones, infos = env.step(actions.detach())
            elif legged_gym_version == 'new':
                rets = env.step(actions.detach())
                obs = rets[0]; rews, dones, infos = rets[-3:]
            rews, dones = rews.cpu(), dones.cpu()

            ''' calculate reward & save data '''
            cur_reward_sum += rews
            cur_episode_length += 1
            
            new_ids = torch.nonzero(dones).reshape(-1)
            if save_data in ['chunk', 'episode']:
                general_valid_mask = (dones & (left_episodes_to_skip == 0))
                if limit_mode is None or limit_mode == 'None':
                    valid_ids = torch.nonzero(
                        general_valid_mask
                    ).reshape(-1)
                elif limit_mode == 'rew':
                    valid_ids = torch.nonzero(
                        general_valid_mask & (cur_reward_sum > reward_limit)
                    ).reshape(-1)
                elif limit_mode == 'ep_len':
                    valid_ids = torch.nonzero(
                        general_valid_mask & (cur_episode_length > eplen_limit)
                    ).reshape(-1)
                else:
                    raise NotImplementedError(f'limit_mode: {limit_mode}, {type(limit_mode)}')

                if valid_ids.numel() > 0:
                    eplen = cur_episode_length[valid_ids].numpy()
                    state = state_buf[valid_ids, :eplen.max() + 10, :].clone().detach().numpy()
                    action = action_buf[valid_ids, :eplen.max() + 10, :].clone().detach().numpy()
                    if save_data == 'chunk':
                        if data_mode == 'nonmdp':
                            src_sp, tgt_sp = convert_episodes_to_samples_nonmdp_numba(
                                state, action, eplen, seq_len, t.n, this_collect_samples,
                                state_dim, action_dim, src_dim, tgt_dim, src_mode, tgt_mode)
                        elif data_mode == 'mdp':
                            src_sp, tgt_sp = convert_episodes_to_samples_mdp_numba(
                                state, action, eplen, t.n, this_collect_samples,
                                state_dim, action_dim, src_dim, tgt_dim, src_mode, tgt_mode)
                        src_data[t.n:t.n + src_sp.shape[0], :, :] = src_sp[:]
                        tgt_data[t.n:t.n + src_sp.shape[0], :, :] = tgt_sp[:]
                        t.update(src_sp.shape[0])
                    elif save_data == 'episode':
                        update_times = 0
                        for valid_id in valid_ids:
                            if not append_episode(state_buf[valid_id, :buf_idx[valid_id], :].clone().detach().numpy(),
                                                    action_buf[valid_id, :buf_idx[valid_id], :].clone().detach().numpy(),
                                                    save_data_info):
                                break
                            else:
                                update_times += 1
                        if update_times > 0:
                            t.update(update_times)
                    new_rews = cur_reward_sum[valid_ids].numpy().tolist()
                    new_lens = cur_episode_length[valid_ids].numpy().tolist()
                else:
                    new_rews, new_lens = [], []
            else:
                raise NotImplementedError()
            
            ''' save rewards & eplens, print info '''
            if len(new_rews) > 0:
                rewbuffer.extend(new_rews)
                lenbuffer.extend(new_lens)
                if not simplify_print_info:
                    print(f'{task} add rews:\n{new_rews}')
                    print(f'{task} add lens:\n{new_lens}')
            
            ''' empty reward & eplen & buf_ids cache '''
            if new_ids.shape[0] > 0:
                cur_reward_sum[new_ids] = 0
                cur_episode_length[new_ids] = 0
                buf_idx[new_ids] = 0
            
            ''' update left_episodes_to_skip '''
            if skip_episodes_valid_flag:
                left_episodes_to_skip -= dones.type(torch.int64)
                left_episodes_to_skip.clamp_(min=0)
                if torch.sum(left_episodes_to_skip) == 0:
                    skip_episodes_valid_flag = False
    # All Play
    else:
        left_play_times = torch.full((num_envs,), play_ep_cnt, dtype=torch.int64)
        estimated_play_steps = int(play_ep_cnt * (max_episode_length + 10))
        sync_flag = torch.zeros((num_envs), dtype=torch.bool)
        pastkv = None
        play_category_print_flag = True
        sum_inference_time = 0
        total_test_steps = 0
        import time
        # ----------------------preparation of print_inference_action_time---------------------------

        if model_name == "fourier_controller":
            print("fourier_policy device", fourier_policy.fc0.weight.device)
            if args.print_inference_action_time:
                fourier_policy = fourier_policy.to("cpu")
            fourier_policy.reset_recur(num_envs, fourier_policy.fc0.weight.device)
            
        if model_name == 'transformer' and kv_cache and \
            'input_data_mode' in chk_config and \
                (chk_config['input_data_mode'] in CHUNK_LOAD_MODES or chk_config['input_data_mode'] == 'chunk'):
            if args.print_inference_action_time:
                transformer_policy = transformer_policy.to("cpu")
        
        if dt_policy_name is not None:
            # for transformer kv cache episode mode
            if chk_config['add_last_action']:
                actions = torch.zeros((num_envs, chk_config['tgt_dim']), dtype=torch.float32, device=env.device)
        if model_name == 'retnet':
            retnet_policy.clear_cache()
        for steps in tqdm(range(estimated_play_steps)):
            ''' model inference '''
            # action inference start
            with torch.inference_mode():
                if model_name == 'transformer' and not kv_cache:
                    if play_category_print_flag:
                        play_category_print_flag = False
                        print('now in transformer | no KVCache')
                    
                    # # nonmdp mode store observation to buffer
                    # if data_mode == 'nonmdp':
                    #     if legged_gym_version == 'old':
                    #         states, attn_mask, tgt_selector = obs_buf.update(obs.detach())
                    #     elif legged_gym_version == 'new':
                    #         states, attn_mask, tgt_selector = \
                    #             obs_buf.update(merge_tensor_dict_to_single_tensor_fixed(obs, env, task).detach().clone())
                    # elif data_mode == 'mdp':
                    #     if legged_gym_version == 'old':
                    #         states = obs.detach()
                    #     elif legged_gym_version == 'new':
                    #         states = merge_tensor_dict_to_single_tensor_fixed(obs, env, task).detach().clone()

                    # # nonmdp mode fetch actions from buffer, mdp mode fetch last actions
                    # actions_las_t = None
                    # if data_mode == 'nonmdp':
                    #     actions_las_t = action_buf.get_action_las_t() # (num_envs, seq_len, action_dim)
                    # elif data_mode == 'mdp':
                    #     actions_las_t = actions

                    # with torch.inference_mode():
                    #     if data_mode == 'nonmdp':
                    #         actions = get_action(policy=transformer_policy, states=states, actions_las_t=actions_las_t,
                    #                             dt_mode=dt_mode, action_dim=action_dim, data_mode=data_mode,
                    #                             attn_mask=attn_mask, tgt_selector=tgt_selector) # (num_envs, action_dim)
                    #     elif data_mode == 'mdp':
                    #         actions = get_action(policy=transformer_policy, states=states, actions_las_t=actions_las_t,
                    #                             dt_mode=dt_mode, action_dim=action_dim, data_mode=data_mode) # (num_envs, action_dim)
                    # actions = actions.detach().clone()

                    # if calc_dt_mlp_loss:
                    #     mlp_actions = mlp_policy(obs)
                    #     print(f"dt_policy_name: {dt_policy_name} loss of mlp and transformer: {(actions - mlp_actions).pow(2).mean()}")

                    # if data_mode == 'nonmdp':
                    #     action_buf.update(actions)
                    # elif data_mode == 'mdp':
                    #     pass
                    pass
                elif model_name == 'mlp':
                    if play_category_print_flag:
                        play_category_print_flag = False
                        print('now in mlp')
                    
                    if legged_gym_version == 'old':
                        actions = mlp_policy(obs.detach())
                    elif legged_gym_version == 'new':
                        actions = mlp_policy(obs)
                elif model_name == 'transformer' and kv_cache and \
                        'input_data_mode' in chk_config and \
                        (chk_config['input_data_mode'] in CHUNK_LOAD_MODES or chk_config['input_data_mode'] == 'chunk'):
                    if play_category_print_flag:
                        play_category_print_flag = False
                        print('now in transformer | with KVCache | chunk mode')
                    obs = torch.unsqueeze(merge_tensor_dict_to_single_tensor_fixed(obs, env, task, dt_mode), dim=1)
                    obs = obs.to(transformer_policy.embed_state.weight.device)
                    
                    if args.print_inference_action_time and steps % seq_len == seq_len - 1:
                        
                        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                            with record_function("model_inference"):
                                actions, pastkv = transformer_policy(obs, use_cache=True, past_key_values=pastkv)
                        
                        # print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
                        for avg in prof.key_averages():
                            sum_inference_time += avg.self_cpu_time_total / 1000  # 转换为毫秒
                        total_test_steps += 1
                        print(f"mean time of inferencing single action: {sum_inference_time / total_test_steps:3f}ms")
                    else:
                        actions, pastkv = transformer_policy(obs, use_cache=True, past_key_values=pastkv)
                        
                    if pastkv[0][0].size(2) >= seq_len:
                        pastkv = None
                    actions = torch.squeeze(actions, dim=1).type(torch.float32)
                elif model_name == 'transformer' and kv_cache and \
                        'input_data_mode' in chk_config and \
                        (chk_config['input_data_mode'] in EPISODE_LOAD_MODES or chk_config['input_data_mode'] == 'episode'):
                    if play_category_print_flag:
                        play_category_print_flag = False
                        print('now in transformer | with KVCache | episode mode')
                    # 初始化pastkv全0
                    if pastkv is None:
                        pastkv = []
                        n_head, seq_len, hidden_size = chk_config['n_head'], chk_config['seq_len'], \
                            chk_config['hidden_size']
                        for _ in range(chk_config['n_layer']):
                            pastkv.append([])
                            for _ in range(2):
                                pastkv[-1].append(
                                    torch.zeros((num_envs, n_head, seq_len - 1, hidden_size // n_head), 
                                                dtype=torch.float32, device=env.device)
                                )
                            pastkv[-1] = tuple(pastkv[-1])
                        pastkv = tuple(pastkv)
                    # reset pastkv of dones env to 0
                    if 'dones' in locals():
                        first_step_ids = torch.nonzero(dones.to(env.device)).reshape(-1)
                        if first_step_ids.numel() > 0:
                            for i in range(len(pastkv)):
                                for j in range(len(pastkv[i])):
                                    pastkv[i][j][first_step_ids, ...] = 0
                    # reset actions of dones env to 0
                    if 'dones' in locals():
                        first_step_ids = torch.nonzero(dones.to(env.device)).reshape(-1)
                        if first_step_ids.numel() > 0:
                            actions[first_step_ids, ...] = 0
                    # transformer policy generate actions
                    obs = merge_tensor_dict_to_single_tensor_fixed(obs, env, task, dt_mode)
                    if chk_config['add_last_action']:
                        obs = torch.cat([obs, actions], dim=-1)
                    obs = torch.unsqueeze(obs, dim=1)
                    actions, pastkv = transformer_policy(obs, use_cache=True, past_key_values=pastkv)
                    actions = torch.squeeze(actions, dim=1).type(torch.float32)
                    # update pastkv
                    if pastkv[0][0].size(2) >= chk_config['seq_len']:
                        pastkv = tuple([tuple([pastkv[i][j][:, :, 1:, :].detach() for j in range(len(pastkv[i]))]) 
                                        for i in range(len(pastkv))])
                elif model_name == 'retnet':
                    if play_category_print_flag:
                        play_category_print_flag = False
                        print('now in retnet')
                    state = torch.unsqueeze(merge_tensor_dict_to_single_tensor_fixed(obs, env, task, dt_mode), dim=1)
                    actions = retnet_policy(state, use_cache=True)
                    if retnet_policy.cache_len >= seq_len:
                        retnet_policy.clear_cache()
                    actions = torch.squeeze(actions, dim=1).type(torch.float32)
                elif model_name == "fourier_controller":
                    obs = torch.unsqueeze(merge_tensor_dict_to_single_tensor_fixed(obs, env, task, dt_mode), dim=1)
                    if steps == 0:
                        print("\n obs", obs.shape)  # (num_envs, 1, state_dim)
                    
                    if chk_config["is_chunk_wise"] == False and steps != 0 and steps % chk_config["seq_len"] == 0:  # (chk_config["seq_len"] - 1)
                        fourier_policy.reset_recur(num_envs, fourier_policy.fc0.weight.device)
                    
                    physical_state = obs[..., :-ctx_dim]
                    ctx = obs[..., -ctx_dim:]
                    physical_state = physical_state.to(fourier_policy.fc0.weight.device)
                    ctx = ctx.to(fourier_policy.fc0.weight.device)
                    
                    if args.print_inference_action_time and steps % seq_len == seq_len - 1:
                        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                            with record_function("model_inference"):
                                actions = fourier_policy.forward(physical_state, ctx)
                        
                        # print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
                        for avg in prof.key_averages():
                            sum_inference_time += avg.self_cpu_time_total / 1000  # 转换为毫秒
                        total_test_steps += 1
                        print(f"mean time of inferencing single action: {sum_inference_time / total_test_steps:3f}ms")
                        # print("avg", sum_inference_time / total_steps)
                    else:
                        actions = fourier_policy.forward(physical_state, ctx)
                        
                    actions = torch.squeeze(actions, dim=1)
                elif model_name == "mlp_imitation":
                    obs = torch.unsqueeze(merge_tensor_dict_to_single_tensor_fixed(obs, env, task, dt_mode), dim=1)
                    if steps == 0:
                        print("\n obs", obs.shape)  # (num_envs, 1, state_dim)
                    actions = mlp_imitation_policy.forward(obs)
                    actions = torch.squeeze(actions, dim=1)
                elif model_name == "rnn_imitation":
                    obs = torch.unsqueeze(merge_tensor_dict_to_single_tensor_fixed(obs, env, task, dt_mode), dim=1)
                    
                    if steps != 0 and steps % chk_config["seq_len"] == 0:  # (chk_config["seq_len"] - 1)
                        rnn_hidden = rnn_imitation_policy.init_hidden(num_envs, device=env.device)
                    
                    if steps == 0:
                        print("\n obs", obs.shape)  # (num_envs, 1, state_dim)
                        rnn_hidden = rnn_imitation_policy.init_hidden(num_envs, device=env.device)
                    actions, rnn_hidden = rnn_imitation_policy.forward(obs, rnn_hidden)
                    actions = torch.squeeze(actions, dim=1)
                
                # calc dt & mlp loss
                if model_name != 'mlp' and calc_dt_mlp_loss:
                    alive_ids = torch.nonzero(~sync_flag).reshape(-1).detach()
                    mlp_actions = mlp_policy(obs)
                    cache_len = 0 if pastkv is None else pastkv[0][0].size(2)
                    loss_mlp_dt = (actions[alive_ids, ...] - mlp_actions[alive_ids, ...]).pow(2).mean()
                    print(f"cache len: {cache_len}, dt_policy_name: {dt_policy_name}, sync_cnt: {alive_ids.numel()}, loss of mlp and transformer: {loss_mlp_dt}")
                    if calc_dt_mlp_loss:
                        mlp_actions = mlp_policy(obs)
                        loss_mlp_dt = (actions - mlp_actions).pow(2).mean()
                        print(f"retnet_len: {retnet_len}, dt_policy_name: {dt_policy_name} loss of mlp and transformer: {loss_mlp_dt:.6f}")

            # action inference end

            ''' env step '''
            if legged_gym_version == 'old':
                obs, _, rews, dones, infos = env.step(actions)
            elif legged_gym_version == 'new':
                rets = env.step(actions)
                obs = rets[0]; rews, dones, infos = rets[-3:]
            rews, dones = rews.cpu(), dones.cpu()

            ''' model post do after env step '''
            if model_name == 'transformer':
                if data_mode == 'nonmdp':
                    obs_buf.post_update(dones)
                    action_buf.post_update(dones)
                elif data_mode == 'mdp':
                    actions[new_ids, :] = 0 # empty actions buffer if this episode ends

            ''' calculate reward '''
            cur_reward_sum += rews
            cur_episode_length += 1
            
            new_ids = torch.nonzero(dones.detach()).reshape(-1)
            valid_ids = torch.nonzero(
                dones.detach() & \
                ~sync_flag & \
                (left_play_times > 0)
            ).reshape(-1)

            if valid_ids.numel() > 0:
                valid_rews = cur_reward_sum[valid_ids].numpy().tolist()
                valid_lens = cur_episode_length[valid_ids].numpy().tolist()
                rewbuffer.extend(valid_rews)
                lenbuffer.extend(valid_lens)
                if not simplify_print_info:
                    print(f'{task} add rews:\n{valid_rews}')
                    print(f'{task} add lens:\n{valid_lens}')
            
            if new_ids.numel() > 0:
                cur_reward_sum[new_ids] = 0
                cur_episode_length[new_ids] = 0
            
            ''' record videos '''
            if args.record:
                this_img_idx = recode_adder.update()
                if this_img_idx is not None:
                    this_img_path = join(img_dir, f"{this_img_idx}.png")
                    env.gym.write_viewer_image_to_file(env.viewer, this_img_path)
            
            ''' sync envs '''
            sync_flag |= dones
            if torch.sum(sync_flag) == num_envs:
                obs, _ = env.reset()
                pastkv, retnet_len = None, 0
                sync_flag[:], cur_reward_sum[:], cur_episode_length[:] = 0, 0, 0
                left_play_times -= 1
                if model_name == "fourier_controller":
                    fourier_policy.reset_recur(num_envs, fourier_policy.fc0.weight.device)
                if model_name == "rnn_imiation":
                    rnn_hidden = rnn_imitation_policy.init_hidden(num_envs, device=env.device)
                if torch.sum(left_play_times) == 0:
                    break
    
    ''' 输出play结果 '''
    print(list(rewbuffer))
    print(list(lenbuffer))
    summary_info = ''
    if model_name == 'transformer':
        summary_info += f'dt_policy_name: {dt_policy_name}'
    summary_info += f' task: {task}, num_envs: {env.num_envs}, \
episodes: {len(rewbuffer)}, \
rew_mean: {statistics.mean(rewbuffer) if len(rewbuffer) > 0 else -999:.3f}, \
len_mean: {statistics.mean(lenbuffer) if len(lenbuffer) > 0 else -999}'

    summary_info += evaluate(list(rewbuffer))[0]

    print(summary_info)

    ''' 合并录制图片为视频 '''
    if args.record:
        create_vedio_from_imgs(img_dir, f'{record_name}.mp4', crf=crf)
        virtual_display.stop()

    if save_data == 'chunk':
        print(f"src_data: mean: {src_data.mean()}, std: {src_data.std()}")
        print(f"tgt_data: mean: {tgt_data.mean()}, std: {tgt_data.std()}")
        if data_mode == 'nonmdp':
            src_shape = (total_collect_samples, seq_len, src_dim)
            tgt_shape = (total_collect_samples, seq_len, tgt_dim)
        elif data_mode == 'mdp':
            src_shape = (total_collect_samples, src_dim, 1)
            tgt_shape = (total_collect_samples, tgt_dim, 1)
        file_format = 'dat'
        data_src_path = join(data_dir, f'src_{src_shape[0]}_{src_shape[1]}_{src_shape[2]}.{file_format}')
        data_tgt_path = join(data_dir, f'tgt_{tgt_shape[0]}_{tgt_shape[1]}_{tgt_shape[2]}.{file_format}')
        src_fp = np.memmap(data_src_path, dtype=np.float32, mode='r+', shape=src_shape)
        tgt_fp = np.memmap(data_tgt_path, dtype=np.float32, mode='r+', shape=tgt_shape)
        this_slice = get_split_slice(total_collect_samples, processors, local_rank)
        print(f'Writing to disk, processors: {processors}, local_rank: {local_rank}, \
total_samples: {total_collect_samples}, this_slice: {this_slice}')
        src_fp[this_slice, :, :] = src_data[:] # :, :, :
        tgt_fp[this_slice, :, :] = tgt_data[:]
        print(f'save chunk data process[{local_rank}] done!')
    elif save_data == 'episode':
        assert save_data_info['now_collect_episodes'] == this_collect_samples, f"{save_data_info['now_collect_episodes']}, {this_collect_samples}"
        save_data_info['state_episode_np'], save_data_info['action_episode_np'], save_data_info['episode_length_np'] = \
            shuffle_episodes(save_data_info['state_episode_np'][:save_data_info['now_collect_steps'], :],
                             save_data_info['action_episode_np'][:save_data_info['now_collect_steps'], :],
                             save_data_info['episode_length_np'])
        file_format = 'dat'
        tgt_data_dir = join(data_dir, 'sub', str(local_rank))
        save_np_mmap(save_data_info['state_episode_np'], tgt_data_dir, 'src', file_format)
        save_np_mmap(save_data_info['action_episode_np'], tgt_data_dir, 'tgt', file_format)
        save_np_mmap(save_data_info['episode_length_np'], tgt_data_dir, 'meta', file_format)
        print(f'save episode data process[{local_rank}] to {data_dir}, \
avg episode length: {np.mean(save_data_info["episode_length_np"])}')
    return

if __name__ == '__main__':
    
    play()
