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
import random
import hydra
from omegaconf import DictConfig, OmegaConf
from FCNet.DT.utils.hydra_utils import *
from FCNet.DT.utils.common import print_dict, omegaconf_to_dict, get_availble_gpus, \
    smart_load_run, parse_dt_mode_to_src_tgt_dim, encode_arg_names_into_arg_list, \
    parse_dt_policy_dir, clear_components, read_np_mmap, save_np_mmap
from FCNet.DT.utils.name_utils import EPISODE_SAVE_DATA_FORMAT

import os
os.environ['HYDRA_FULL_ERROR'] = '1'
from os.path import join
import shutil
import json
import pprint
import multiprocessing as mp
import numpy as np
import gc
from tqdm import tqdm

def save_data_func(arg):
    local_rank, arg_names = arg[-2], arg[-1]
    assert len(arg_names) + 2 == len(arg), \
        f"arg_names len: {len(arg_names)}, arg len: {len(arg)}"
    cmd = f"DISPLAY="" python ./sub/play_sub.py"
    for i, arg_name in enumerate(arg_names):
        if arg_name in ['headless', 'resume', 'push_robots', 'add_noise', 'simplify_print_info', 'dummy']:
            if arg[i]: cmd += f' --{arg_name}'
        elif arg_name in ['availble_gpus']:
            cmd += f' --device_id {arg[i][local_rank]}'
        else:
            cmd += f' --{arg_name} {arg[i]}'
    cmd += f' --local_rank {local_rank}'
    # print(f'cmd={cmd}')
    os.system(cmd)
    return

availble_gpus = get_availble_gpus(max_used_memory=3000)
print(f'availble_gpus: {availble_gpus}')
assert len(availble_gpus) > 0

@hydra.main(version_base=None, config_path="../../cfg", config_name="config")
def main(cfg: DictConfig):
    global availble_gpus
    # yaml config 转为 dict
    cfg_dict = omegaconf_to_dict(cfg)
    # print_dict(cfg_dict)
    # yaml 参数
    
    task = cfg_dict['task_name']
    local_rank = cfg_dict['local_rank']
    headless = cfg_dict['headless']
    # local_rank = cfg_dict['local_rank']
    record = cfg_dict['record']
    play_ep_cnt = cfg_dict['play_ep_cnt']
    ac_ratio = cfg_dict['ac_ratio']
    crf = cfg_dict['crf']
    num_envs = cfg_dict['num_envs']
    seq_len = cfg_dict['seq_len']
    limit_mode = cfg_dict['limit_mode_name']
    dt_mode = cfg_dict['dt_mode_name']
    resume = cfg_dict['resume']
    load_run = cfg_dict['load_run']
    checkpoint = cfg_dict['checkpoint']
    max_workers = cfg_dict['max_workers']
    dummy = cfg_dict['dummy']
    push_robots = cfg_dict['push_robots']
    add_noise = cfg_dict['add_noise']
    kv_cache = cfg_dict['kv_cache']
    simplify_print_info = cfg_dict['simplify_print_info']
    data_mode = cfg_dict['data_mode_checked']
    save_data = cfg_dict['save_data']
    multi_gpu = cfg_dict['multi_gpu']
    calc_dt_mlp_loss = cfg_dict['calc_dt_mlp_loss']
    legged_gym_version = cfg_dict['legged_gym_version_checked']
    dt_policy_name = cfg_dict['dt_policy_name']
    model_name = cfg_dict['model_name_checked']
    data_save_to = cfg_dict['data_save_to']
    remerge_sub_data = cfg_dict['remerge_sub_data'] # don't sample data, only merge subdatas
    skip_episodes = cfg_dict['skip_episodes'] # episodes to skip, only valid in save data
    aligo_auto_complete_chks = cfg_dict['aligo_auto_complete_chks'] # fetch chks from aligo if not found locally
    
    log_dir_name = cfg_dict['task']['log_dir_name']
    total_collect_samples = cfg_dict['task']['total_collect_samples']
    max_episode_length = cfg_dict['task']['max_episode_length']
    state_dim = cfg_dict['task']['state_dim']
    action_dim = cfg_dict['task']['action_dim']
    reward_limit = cfg_dict['task']['reward_limit']
    
    print_inference_action_time = cfg_dict['print_inference_action_time'] if 'print_inference_action_time' in cfg_dict else False
    print("print_inference_action_time", print_inference_action_time)
    print(cfg_dict)
    
    legged_gym_root_dir = None
    if legged_gym_version == 'old':
        from legged_gym import LEGGED_GYM_ROOT_DIR
        legged_gym_root_dir = LEGGED_GYM_ROOT_DIR
    elif legged_gym_version == 'new':
        from legged_robot_personal import LEGGED_ROBOT_PERSONAL_ROOT_DIR
        legged_gym_root_dir = LEGGED_ROBOT_PERSONAL_ROOT_DIR
    
    if resume:
        load_run, checkpoint = smart_load_run(join(legged_gym_root_dir, 'logs', log_dir_name), load_run, checkpoint)
    
    if max_workers is not None and len(availble_gpus) > max_workers:
        availble_gpus = availble_gpus[:max_workers]
    
    device_id = availble_gpus[local_rank] if multi_gpu else availble_gpus[0]
    
    if save_data is not None:
        # set default episode length limit
        processors = len(availble_gpus)
        if save_data == 'chunk':
            if data_mode == 'mdp':
                eplen_limit = 3
            elif data_mode == 'nonmdp':
                eplen_limit = seq_len + 10
        
        # create/clear save data dir
        if save_data == 'chunk':
            raise NotImplementedError
            data_dir_name = f'{task}_{seq_len if data_mode == "nonmdp" else "mdp"}_{dt_mode}_{save_data}'
        elif save_data == 'episode':
            # no need for dt_mode & seq_len
            # data_dir_name = f'{task}_{save_data}_{total_collect_samples}_{max_episode_length}'
            player_level = 'expert'
            data_dir_name = eval(EPISODE_SAVE_DATA_FORMAT)
            print(f'episode data_dir_name: {data_dir_name}')
        if limit_mode is not None:
            data_dir_name += '_{}{}'.format(
                'r' if limit_mode == 'rew' else 'l',
                int(reward_limit) if limit_mode == 'rew' else int(eplen_limit),
            )
        data_dir = join(data_save_to, data_dir_name)
        if not remerge_sub_data:
            clear_components(data_dir)
        
        # parse data's src_dim and tgt_dim from dt_mode[s_a,...], state_dim and action_dim
        src_dim, tgt_dim = parse_dt_mode_to_src_tgt_dim(dt_mode, state_dim, action_dim)

        # prepare np.memmap or sub_dir before running simulator to collect data
        if save_data == 'chunk':
            raise NotImplementedError
            if data_mode == 'nonmdp':
                src_shape = (total_collect_samples, seq_len, src_dim)
                tgt_shape = (total_collect_samples, seq_len, tgt_dim)
            elif data_mode == 'mdp':
                src_shape = (total_collect_samples, src_dim, 1)
                tgt_shape = (total_collect_samples, tgt_dim, 1)

            data_src_path = join(data_dir, 'src_{}_{}_{}.dat'.format(*src_shape))
            data_tgt_path = join(data_dir, 'tgt_{}_{}_{}.dat'.format(*tgt_shape))

            src_fp = np.memmap(data_src_path, dtype=np.float32, mode='w+', shape=src_shape)
            tgt_fp = np.memmap(data_tgt_path, dtype=np.float32, mode='w+', shape=tgt_shape)
            del src_fp, tgt_fp
            gc.collect()
        elif save_data == 'episode':
            if not remerge_sub_data:
                os.makedirs(join(data_dir, 'sub'))
                for local_rank in range(processors):
                    os.makedirs(join(data_dir, 'sub', str(local_rank)))
        
        # run simulator to collect data
        # pass_arg_vars(list) + [local_rank, var_names] = arg
        if not remerge_sub_data:
            arg_names = ['data_dir', 'availble_gpus', 'processors', 
                'task', 'dt_mode', 'limit_mode', 'headless', 'load_run', 'add_noise', 
                'push_robots', 'resume', 'num_envs', 'load_run', 'checkpoint',
                'total_collect_samples', 'reward_limit', 'legged_gym_version', 
                'model_name', 'save_data', 'data_mode', 'skip_episodes',
                'simplify_print_info', 'dummy', 'max_episode_length']
            if save_data == 'chunk':
                raise NotImplementedError
                arg_names += ['seq_len', 'eplen_limit']
            pool = mp.Pool(processes=processors)
            pool.map(save_data_func,
                    encode_arg_names_into_arg_list(locals(), globals(), processors, arg_names))
            pool.close(); pool.join()
        
        # # 逐个shuffle子进程收集的数据，节约内存(但是会多一次读取，一次写入)
        # def shuffle_episodes_from_np(data_dir: str):
        #     print('start reading data!')
        #     fpS: List[np.memmap] = []
        #     for file_name, dtype in [('src', np.float32), ('tgt', np.float32), ('meta', np.int32)]:
        #         fpS.append(read_np_mmap(data_dir, file_name, 'dat', dtype, 'r+'))
            
        #     dataS = list(map(np.zeros_like, fpS))
        #     for i in range(len(dataS)):
        #         dataS[i][:] = fpS[i][:]
        #     print('start shuffling data!')
        #     x, y, z = list(map(np.zeros_like, fpS))
        #     acc_eplen, slices = 0, []
        #     for eplen in dataS[2]:
        #         slices.append((eplen, slice(acc_eplen, acc_eplen + eplen)))
        #         acc_eplen += eplen
        #     random.shuffle(slices)
        #     acc_eplen = 0
        #     for i, (eplen, epslice) in enumerate(slices):
        #         x[acc_eplen: acc_eplen + eplen, :] = dataS[0][epslice, :]
        #         y[acc_eplen: acc_eplen + eplen, :] = dataS[1][epslice, :]
        #         z[i] = eplen
        #         acc_eplen += eplen
        #     print('start saving data!')
        #     for fp, mem in zip(fpS, [x, y, z]):
        #         fp[:] = mem[:]
        #         fp.flush()
        #     del fpS, x, y, z, dataS
        # print('start shuffling data!')
        # for i in tqdm(range(processors)):
        #     shuffle_episodes_from_np(join(data_dir, 'sub', str(i)))
        
        # data post-processing
        if save_data == 'chunk':
            pass
        elif save_data == 'episode':
            # 读取子程序保存的数据，并且合并起来保存，删除掉子程序的数据
            print('start reading sub data and merge in memory!')
            final_data = [None, None, None] # src tgt meta
            for i in tqdm(range(processors)):
                def merge2finalData(final_data: List[np.ndarray], fpS: List[np.memmap]):
                    for i, (_, fp) in enumerate(zip(final_data, fpS)):
                        if final_data[i] is None:
                            final_data[i] = np.zeros(fp.shape, dtype=fp.dtype)
                            final_data[i][:] = fp[:]
                        else:
                            final_data[i] = np.concatenate([final_data[i], fp[:]], axis=0)
                fpS = []
                for file_name, dtype in [('src', np.float32), ('tgt', np.float32), ('meta', np.int32)]:
                    fpS.append(read_np_mmap(join(data_dir, 'sub', str(i)), file_name, 'dat', dtype, 'r'))
                merge2finalData(final_data, fpS)
                del fpS
            
            # 先删除子进程保存的数据
            print('start deleting sub data!')
            shutil.rmtree(join(data_dir, 'sub'), ignore_errors=True)
            
            # 再把内存中的数据保存到磁盘
            print('start save final data to the disk!')
            for i, file_name in enumerate(['src', 'tgt', 'meta']):
                save_np_mmap(final_data[i], data_dir, file_name, 'dat')
    
    elif dt_policy_name is not None:
        # 策略评估
        print("play print_inference_action_time", print_inference_action_time)
        
        cmd = f"DISPLAY='' python ./sub/play_sub.py --task {task} \
--dt_policy_name {dt_policy_name} \
--load_run {load_run} \
--checkpoint {checkpoint} --legged_gym_version {legged_gym_version} \
--device_id {device_id} --play_ep_cnt {play_ep_cnt} --ac_ratio {ac_ratio} \
--num_envs {num_envs} \
--crf {crf}"
        if print_inference_action_time: cmd += ' --print_inference_action_time'
        if calc_dt_mlp_loss: cmd += ' --calc_dt_mlp_loss'
    elif dt_policy_name is None:
        cmd = f'DISPLAY="" python ./sub/play_sub.py --task {task} \
--load_run {load_run} \
--checkpoint {checkpoint} --device_id {device_id} \
--play_ep_cnt {play_ep_cnt} --ac_ratio {ac_ratio} \
--num_envs {num_envs} --crf {crf} \
--legged_gym_version {legged_gym_version} --model_name {model_name}'
    else:
        raise NotImplementedError('Do not know what to run!')
    if 'cmd' in locals():
        if headless: cmd += ' --headless'
        if resume: cmd += ' --resume'
        if record: cmd += ' --record'
        if add_noise: cmd += ' --add_noise'
        if push_robots: cmd += ' --push_robots'
        if simplify_print_info: cmd += ' --simplify_print_info'
        if dummy: cmd += ' --dummy'
        if kv_cache: cmd += ' --kv_cache'
        if aligo_auto_complete_chks: cmd += ' --aligo_auto_complete_chks'
        cmd += f' --max_episode_length {int(max_episode_length)}'
        os.system(cmd)
    return

if __name__ == '__main__':
    main()
    print('ALL DONE!')
