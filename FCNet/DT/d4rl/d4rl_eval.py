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
import os
import re
from os.path import join, exists
import shutil
import json
import pickle
import lzma
import statistics
import random

import gym
import gymnasium
import numpy as np
import torch

import d4rl
from d4rl.ope import normalize
from d4rl import infos

import numpy as np
import torch
torch.set_printoptions(sci_mode=False, linewidth=100_000)

from collections import deque
import statistics
import time
import argparse
from tqdm import tqdm
import pprint
pp = pprint.PrettyPrinter()

from FCNet.DT.utils.common import get_parameter_number, merge_tensor_dict_to_single_tensor_fixed


def get_args():
    parser = argparse.ArgumentParser()
    custom_parameters = [
        {"name": "--train_task", "type": str, "default": "ant", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--eval_task", "type": str, "default": "ant", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},

        # dt params
        {"name": "--dt_policy_name", "type": str, "default": None},
        {"name": "--device_id", "type": int, "default": None, "help": ""},
        # {"name": "--export_model_as_jit", "action": "store_true", "default": False},
        # {"name": "--use_fp16", "action": "store_true", "default": False},
        # {"name": "--use_flash_attn", "action": "store_true", "default": False},
        # {"name": "--double_v_dim", "action": "store_true", "default": False},
        # {"name": "--is_causal", "action": "store_true", "default": False},
        {"name": "--kv_cache", "action": "store_true", "default": False},
    ]
    for argument in custom_parameters:
        if ("name" in argument) and ("type" in argument or "action" in argument):
            help_str = ""
            if "help" in argument:
                help_str = argument["help"]

            if "type" in argument:
                if "default" in argument:
                    parser.add_argument(argument["name"], type=argument["type"], default=argument["default"], help=help_str)
                else:
                    parser.add_argument(argument["name"], type=argument["type"], help=help_str)
            elif "action" in argument:
                parser.add_argument(argument["name"], action=argument["action"], help=help_str)

        else:
            print()
            print("ERROR: command line argument name, type/action must be defined, argument not added to parser")
            print("supported keys: name, type, default, action, help")
            print()

    args = parser.parse_args()
    
    return args


device = "cuda"


def eval_policy(policy, env_name, seed, chk_config, eval_episodes=10):
    # env_name is like "halfcheetah-medium-expert-v2"
    policy_id = env_name[:-3]
    # policy_id is like "halfcheetah-medium-expert", as argument of normalize()
    key = policy_id + '-v0'
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    
    ctx_dim = chk_config["ctx_dim"]
    model_name = chk_config["model_name"]
    dt_mode = chk_config["dt_mode"]
    
    avg_reward = 0.
    state_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]
    src_dim = chk_config["src_dim"]  # state_dim of model: chk_config["src_dim"]
    tgt_dim = chk_config["tgt_dim"]  # action_dim of model: chk_config["tgt_dim"]
    
    play_category_print_flag = True
    
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False  # state: (state_dim,)
        steps = 0
        # return_to_go 1.15 for medium-expert and expert; 1.0 for medium and medium-replay
        include_expert = "expert" in env_name
        is_medium = policy_id[-6:] == "medium"
        # return_to_go = 1.15 if include_expert else 1.0  # standard setting TODO
        return_to_go = 1.2 if include_expert else 0.8  # standard setting for Adroit
        # is_halfcheetah = "halfcheetah" in env_name
        # if is_halfcheetah:
        #     return_to_go = 0.9
        # else:
        #     return_to_go = 1.05
        action = np.zeros(action_dim)
        pastkv = None
        # init
        if model_name == "fourier_controller":
            # print("fourier_policy device", policy.fc0.weight.device)
            policy = policy.to(device)
            policy.reset_recur(1, policy.fc0.weight.device)
        if model_name == 'retnet':
            policy.clear_cache()
        
        while not done:
            with torch.inference_mode():
                state = torch.FloatTensor(state)
                action = torch.FloatTensor(action)
                # ----------above is raw state and action----------
                
                # concat return_to_go and state
                state = torch.cat([torch.FloatTensor([return_to_go]), state], dim=-1)
                if dt_mode == "as_a":
                    state = torch.concat([action, state], dim=-1)
                if state.shape[0] < src_dim:
                    # concat 0
                    state = torch.cat([state, torch.zeros(src_dim - state.shape[0])], dim=-1)
                    
                state = state.to(device)
                state = state.unsqueeze(0).unsqueeze(0)# (1, 1, src_dim)
                if model_name == "fourier_controller":
                    if chk_config["is_chunk_wise"] == False and steps != 0 and steps % chk_config["seq_len"] == 0:  # (chk_config["seq_len"] - 1)
                        policy.reset_recur(1, policy.fc0.weight.device)
                        if steps <= chk_config["seq_len"]:
                            return_to_go = 1.15 if include_expert else 1.0  # standard setting
                        
                    physical_state = state[..., :-ctx_dim]
                    ctx = state[..., -ctx_dim:]
                    action = policy.forward(physical_state, ctx)
                    # print("action.shape", action.shape)
                    action = action[0, 0, :action_dim]  # (1, 1, tgt_dim) -> (action_dim,)
                elif model_name == "transformer":
                    if play_category_print_flag:
                        play_category_print_flag = False
                        print('now in retnet')
                    action, pastkv = policy(state, use_cache=True, past_key_values=pastkv)
                    if pastkv[0][0].size(2) >= chk_config["seq_len"]:
                        pastkv = None
                    action = action[0, 0, :action_dim]  # (1, 1, tgt_dim) -> (action_dim,)
                elif model_name == 'retnet':
                    if play_category_print_flag:
                        play_category_print_flag = False
                        print('now in retnet')
                    action = policy(state, use_cache=True)
                    if policy.cache_len >= chk_config["seq_len"]:
                        policy.clear_cache()
                    action = action[0, 0, :action_dim]  # (1, 1, tgt_dim) -> (action_dim,)
                action = action.cpu().detach()
                # ----------below is raw state and action----------
            action = action.numpy()
            state, reward, done, _ = eval_env.step(action)
            return_to_go -= reward / (infos.REF_MAX_SCORE[key] - infos.REF_MIN_SCORE[key])
            avg_reward += reward
            steps += 1

    avg_reward /= eval_episodes
    normalized_reward = normalize(policy_id, avg_reward) * 100.0

    print(f"Evaluation over {eval_episodes} episodes, avg_reward: {avg_reward:.3f} normalized reward: {normalized_reward:.3f}")
    return eval_episodes, avg_reward, normalized_reward

def main(args=get_args()):
    train_task = args.train_task  
    # train_task is like "ant" or "ant-halfcheetah-hopper-unitree_general-walker2d"
    eval_task = args.eval_task
    # eval_task is like "ant-medium-expert-v2" or "halfcheetah-medium-expert-v2"
    dt_policy_name: str = args.dt_policy_name
    print("dt_policy_name", dt_policy_name)
    print("train_task", train_task)
    print("eval_task", eval_task)
    if dt_policy_name is not None:
        local_dt_policy_dir = join('../log', train_task, dt_policy_name)
        
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
    
        # print("chk_config", chk_config)
        print("chk_config")
        pp.pprint(chk_config)
        dt_mode = chk_config['dt_mode']
    
    # ref to play_sub.py
    print("model_name", model_name)
    ''' play dt 加载模型 '''
    if model_name == 'transformer':
        if chk_config['export_model_as_jit']:
            print(f"load exported jit model.")
            policy = torch.jit.load(join(local_dt_policy_dir, 'exported_jit_model.pt'),
                                                map_location='cpu')
        else:
            from FCNet.DT.models import DecisionTransformer
            policy = DecisionTransformer(chk_config)
            get_parameter_number(policy)
            policy.load_state_dict(checkpoint['model'])
        print("transformer created")
        policy = policy.to(device)
        policy.eval()
    elif model_name == 'retnet':
        from FCNet.DT.models import DecisionRetNet
        policy = DecisionRetNet(chk_config)
        get_parameter_number(policy)
        policy.load_state_dict(checkpoint['model'])
        print("retnet created")
        policy = policy.to(device)
        policy.eval()
    elif model_name == 'fourier_controller':
        from FCNet.DT.models.fourier_controller import FourierController
        policy = FourierController(chk_config)
        print("fourier_policy created")
        get_parameter_number(policy)
        policy.load_state_dict(checkpoint['model'])
        policy = policy.to(device)
        policy.eval()
    elif model_name == 'mlp_imitation':
        from FCNet.DT.models.mlp_imitation import MLPImitationModel
        policy = MLPImitationModel(chk_config)
        get_parameter_number(policy)
        policy.load_state_dict(checkpoint['model'])
        policy = policy.to(device)
        policy.eval()
    elif model_name == 'rnn_imitation':
        from FCNet.DT.models.rnn_imitation import RNNImitationModel
        policy = RNNImitationModel(chk_config)
        get_parameter_number(policy)
        policy.load_state_dict(checkpoint['model'])
        policy = policy.to(device)
        policy.eval()

    # assert model_name == "fourier_controller", "only support fourier_controller now"
    seed_num = 3
    normalized_reward_list = []
    for seed in range(seed_num):
        seed = np.random.randint(0, 10000)
        eval_episodes, avg_reward, normalized_reward = eval_policy(
            policy, eval_task, 
            seed, chk_config, eval_episodes=10)
        normalized_reward_list.append(normalized_reward)
    print("---------------------------------------")
    print(f"Evaluation over {seed_num} seeds, avg_normalized_reward+-std: {statistics.mean(normalized_reward_list):.3f}+-{statistics.stdev(normalized_reward_list):.3f}")
    print("---------------------------------------")
    
if __name__ == "__main__":
    main()