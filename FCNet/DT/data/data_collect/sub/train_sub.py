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

import numpy as np
import os
from datetime import datetime

import isaacgym
from isaacgym import gymutil

from legged_gym.envs import *
from legged_gym.utils import task_registry
import torch

from FCNet.DT.utils.common import get_availble_gpus, class_to_dict

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
        
        {"name": "--log_dir", "type": str, "help": ""},
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"

    # pp.pprint(vars(args))
    
    ''' 获取和设置可用GPU '''
    availble_gpus = get_availble_gpus()
    assert len(availble_gpus) > 0
    device_id = availble_gpus[0]
    device = f'cuda:{device_id}'
    
    args.compute_device_id = device_id
    args.graphics_device_id = device_id
    args.sim_device_id = device_id
    args.rl_device = device
    args.sim_device = device
    
    return args

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    pp.pprint(class_to_dict(env_cfg))
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    pp.pprint(class_to_dict(train_cfg))
    
    num_envs = env_cfg.env.num_envs
    num_obs = env_cfg.env.num_observations
    num_act = env_cfg.env.num_actions
    print(f"{args.task}    num_envs: {num_envs}, num_obs: {num_obs}, num_act: {num_act}, seed: {env_cfg.seed}, num_rows: {env_cfg.terrain.num_rows}, num_cols: {env_cfg.terrain.num_cols}, curriculum: {env_cfg.terrain.curriculum}, add_noise: {env_cfg.noise.add_noise}, randomize_friction: {env_cfg.domain_rand.randomize_friction}, push_robots: {env_cfg.domain_rand.push_robots}")
    
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
