from typing import Dict, List, Union, Optional
import re
import os
from os.path import join
import shutil
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

import platform
import threading

from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import numba
import numpy as np
import pyvirtualdisplay
from pyvirtualdisplay.smartdisplay import SmartDisplay

from pprint import PrettyPrinter
pp = PrettyPrinter()

import time

class TimeCounter:
    def __init__(self, train_info: dict, key: str) -> None:
        self.train_info = train_info
        self.key = key
    
    def __enter__(self):
        self._start = time.time()
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        self._end = time.time()
        self.train_info[self.key] = self._end - self._start

def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.match(pattern, f):
            os.remove(os.path.join(dir, f))

def find_file(dir, pattern):
    for f in os.listdir(dir):
        if re.match(pattern, f):
            return f
    return None

def chunk_into_slices(n:int, m:int) -> List[slice]:
    res = []
    base_num = n // m
    extra_num = n - base_num * m
    p = 0
    for i in range(extra_num):
        res.append(slice(p, p+base_num+1))
        p += (base_num + 1)
    for i in range(extra_num, m):
        res.append(slice(p, p+base_num))
        p += base_num
    assert len(res) == m
    return res

def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

def convert_second_to_format(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    h, m, s = int(h), int(m), int(s)
    # return '%d:%02d:%02d' % (h, m, s)
    return '{:d}:{:0>2d}:{:0>2d}'.format(h, m, s)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

def make_std_mask(tgt:torch.Tensor, pad):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
        tgt_mask.data
    )
    return tgt_mask

def loss_calc(out:torch.Tensor, tgt:torch.Tensor):
    '''
        Transformer:
            out, tgt: (bz, predict_len, patch_num*tick_dim)
    '''
    result = dict()
    # loss = (out - tgt).pow(2).mean()
    loss = F.mse_loss(out, tgt, reduction='mean')
    result['loss'] = loss
    return result

def parse(line,qargs):
    numberic_args = ['memory.free', 'memory.total', 'power.draw', 'power.limit']#可计数的参数
    power_manage_enable=lambda v:(not 'Not Support' in v)#lambda表达式，显卡是否滋瓷power management（笔记本可能不滋瓷）
    to_numberic=lambda v:float(v.upper().strip().replace('MIB','').replace('W',''))#带单位字符串去掉单位
    process = lambda k,v:((int(to_numberic(v)) if power_manage_enable(v) else 1) if k in numberic_args else v.strip())
    return {k:process(k,v) for k,v in zip(qargs,line.strip().split(','))}

def query_gpu():
    qargs =['index', 'gpu_name', 'memory.free', 'memory.total', 'power.draw', 'power.limit']
    cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))
    results = os.popen(cmd).readlines()
    return [parse(line,qargs) for line in results]

def get_availble_gpus(max_used_memory=1000):
    gpu_infos = query_gpu()
    availble_gpus = list()
    # pp.pprint(gpu_infos)
    for gpu_info in gpu_infos:
        if gpu_info['memory.total'] - gpu_info['memory.free'] < max_used_memory:
            availble_gpus.append(int(gpu_info['index']))
    return availble_gpus

class RocordAdder:
    def __init__(self, ac_ratio):
        self.img_idx = int(0)
        self.ac_counter = 0
        self.ac_unit = 1.0 / ac_ratio
        return
    def update(self):
        self.ac_counter += self.ac_unit
        diff = int(int(self.ac_counter) - self.img_idx)
        assert diff <= 1
        self.img_idx += diff
        if diff > 0:
            return self.img_idx
        return None


class Compress_Pic_or_Video(object):
    def __init__(self, filePath, inputName, outName, crf=23):
        # outName = "new_" + inputName
        self.filePath = filePath  # 文件地址
        self.inputName = inputName  # 输入的文件名字
        self.outName = outName  # 输出的文件名字
        self.crf = crf
        self.system_ = platform.platform().split("-", 1)[0]
        if self.system_ == "Windows":
            self.filePath = (self.filePath + "\\") if self.filePath.rsplit("\\", 1)[-1] else self.filePath
        elif self.system_ == "Linux":
            self.filePath = (self.filePath + "/") if self.filePath.rsplit("/", 1)[-1] else self.filePath
        self.fileInputPath = self.filePath + inputName
        self.fileOutPath = self.filePath + outName
 
    @property
    def is_video(self):
        videoSuffixSet = {"WMV", "ASF", "ASX", "RM", "RMVB", "MP4", "3GP", "MOV", "M4V", "AVI", "DAT", "MKV", "FIV",
                          "VOB"}
        suffix = self.fileInputPath.rsplit(".", 1)[-1].upper()
        if suffix in videoSuffixSet:
            return True
        else:
            return False
 
    def SaveVideo(self):
        fpsize = os.path.getsize(self.fileInputPath) / 1024
        if fpsize >= 150.0:  # 大于150KB的视频需要压缩
            if self.outName:
                compress = f"ffmpeg -i {self.fileInputPath} -r 10 -pix_fmt yuv420p -vcodec libx264 -preset veryslow -profile:v baseline  -crf {self.crf} -acodec aac -b:a 32k -strict -5 {self.fileOutPath}"
                isRun = os.system(compress)
            else:
                compress = f"ffmpeg -i {self.fileInputPath} -r 10 -pix_fmt yuv420p -vcodec libx264 -preset veryslow -profile:v baseline  -crf {self.crf} -acodec aac -b:a 32k -strict -5 {self.fileInputPath}"
                isRun = os.system(compress)
            if isRun != 0:
                return (isRun, "没有安装ffmpeg")
            return True
        else:
            return True
 
    def Compress_Video(self):
        # 异步保存打开下面的代码，注释同步保存的代码
        thr = threading.Thread(target=self.SaveVideo)
        thr.start()
        thr.join()
        print(f"Compress_Video thread END!")
        return
        # 下面为同步代码
        # fpsize = os.path.getsize(self.fileInputPath) / 1024
        # if fpsize >= 150.0:  # 大于150KB的视频需要压缩
        #     compress = f"ffmpeg -i {self.fileInputPath} -r 10 -pix_fmt yuv420p -vcodec libx264 -preset veryslow -profile:v baseline  -crf {self.crf} -acodec aac -b:a 32k -strict -5 {self.fileOutPath}"
        #     isRun = os.system(compress)
        #     if isRun != 0:
        #         return (isRun, "没有安装ffmpeg")
        #     return True
        # else:
        #     return True

def create_vedio_from_imgs(img_dir, video_name='video.mp4', crf=23):
    print('Merging imgs to vedio!')
    ''' imageio v3 part '''
    import imageio.v3 as iio
    fps = 60
    first_video_name = 'first.mp4'
    video_path = os.path.join(img_dir, first_video_name)
    img_paths = [os.path.join(img_dir, img_name)
        for img_name in os.listdir(img_dir)]
    img_paths.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    t = tqdm(
        enumerate(img_paths),
        total=len(img_paths),
        leave=True,
        ncols=150,
        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
        mininterval=5,
    )
    imgs = []
    for i, img_path in t:
        imgs.append(iio.imread(img_path))
    iio.imwrite(video_path, imgs, fps=fps)
    # iio.imwrite(video_path, imgs, duration=1000/fps)
    
    ''' imageio v2 part '''
    # import imageio.v2 as iio
    # video_name = 'video.mp4'
    # fps = 60
    # video_path = os.path.join(img_dir, video_name)
    # img_paths = [os.path.join(img_dir, img_name)
    #     for img_name in os.listdir(img_dir)]
    # img_paths.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    # video = iio.get_writer(
    #     video_path, format='ffmpeg', mode='I', fps=fps, 
    #     codec='libx264', pixelformat='yuv420p'
    # )
    # t = tqdm(
    #     enumerate(img_paths),
    #     total=len(img_paths),
    #     leave=True,
    #     ncols=150,
    #     bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
    #     mininterval=5,)
    # for i, img_path in t:
    #     frame = iio.imread(img_path)
    #     video.append_data(frame)
    # video.close()

    ''' cv2 part (can't display in vscode) '''
    # import cv2
    # fps = 60
    # width = 1920
    # height = 1080
    # img_paths = [os.path.join(img_dir, img_name) for img_name in os.listdir(img_dir)]
    # img_paths.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    # # myfourcc, video_name = cv2.VideoWriter_fourcc(*'h264'), 'video.mp4'
    # # myfourcc, video_name = cv2.VideoWriter_fourcc(*"mp4v"), 'video.mp4'
    # myfourcc, video_name = cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 'video.mp4'
    # # myfourcc, video_name = cv2.VideoWriter_fourcc(*'XVID'), 'video.avi'
    # video = cv2.VideoWriter(
    #     os.path.join(img_dir, video_name),
    #     myfourcc,
    #     fps,
    #     (width, height),
    # )
    # for img_path in img_paths:
    #     print(f'solve {img_path}')
    #     img = cv2.imread(img_path)
    #     img = cv2.resize(img, (width, height))
    #     video.write(img)
    # video.release()

    for img_path in img_paths:
        os.remove(img_path)
    
    # 压缩视频
    if os.path.exists(video_path):
        print(f'Compressing video at {video_path}')
        savevideo = Compress_Pic_or_Video(img_dir, first_video_name, video_name, crf=crf)
        print(savevideo.Compress_Video())
        os.remove(os.path.join(img_dir, first_video_name))
    print(f"video: {video_name} DONE!")
    return

def class_to_dict(obj) -> dict:
    if not hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def omegaconf_to_dict(d: DictConfig)->Dict:
    """Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation."""
    ret = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        else:
            ret[k] = v
    return ret

def print_dict(val, nesting: int = -4, start: bool = True):
    """Outputs a nested dictionory."""
    if type(val) == dict:
        if not start:
            print('')
        nesting += 4
        for k in val:
            print(nesting * ' ', end='')
            print(k, end=': ')
            print_dict(val[k], nesting, start=False)
    else:
        print(val)

def multi_dataLoader_iter(data_loaders:list):
    for data_loader in data_loaders:
        for src, tgt in data_loader:
            yield src, tgt

def check_load_run_name_valid(x:str):
    "valid example: Aug15_23-49-00_"
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    y = re.split(r'[_-]', x)
    valid = True
    valid &= (len(y) == 5)
    if not valid: return False
    valid &= (len(y[0]) == 5)
    if not valid: return False
    valid &= (y[0][:3] in months)
    if not valid: return False
    valid &= (y[0][3:5].isdigit())
    if not valid: return False
    valid &= (y[1].isdigit())
    if not valid: return False
    valid &= (y[2].isdigit())
    if not valid: return False
    valid &= (y[3].isdigit())
    if not valid: return False
    valid &= (len(y[4]) == 0)
    return valid

def filter_load_run_dirs(run_dirs:list):
    "run_dirs: [Aug15_23-49-00_, ...]"
    return list(filter(check_load_run_name_valid, run_dirs))

def smart_load_run(log_runs_dir:str, load_run, checkpoint):
    '''
        log_runs_dir: join(LEGGED_GYM_ROOT_DIR, 'logs', experiment_name)
        load_run: Aug15_23-49-00_ example
        checkpoint: 1000 example
    '''
    # assert not ((load_run is None) ^ (checkpoint is None)), f'load_run: {load_run}, checkpoint: {checkpoint}'
    if load_run is None:
        run_dirs = os.listdir(log_runs_dir)
        assert len(run_dirs) > 0, f"log_runs_dir: {log_runs_dir} is empty."
        run_dirs = filter_load_run_dirs(run_dirs)
        run_dirs = [(run_dir, os.path.getatime(os.path.join(log_runs_dir, run_dir))) for run_dir in run_dirs]
        run_dirs.sort(key=lambda x: x[1], reverse=True)
        load_run = run_dirs[0][0]
    
    if checkpoint is None:
        chks = os.listdir(os.path.join(log_runs_dir, load_run))
        chks = list(filter(lambda x: '.pt' in x, chks))
        chks.sort(key=lambda x: int(x.split('.')[0].split('_')[1]), reverse=True)
        checkpoint = int(chks[0].split('.')[0].split('_')[1])
    return str(load_run), str(checkpoint)

@numba.njit
def get_split_sz(n, m, idx):
    assert idx < m
    more = n % m
    res = n // m
    res += (1 if idx < more else 0)
    return res

def get_split_range(n, m, idx):
    assert idx < m
    l = 0
    for i in range(idx):
        l += get_split_sz(n, m, i)
    r = l + get_split_sz(n, m, idx)
    return l, r

def get_split_slice(n, m, idx):
    assert idx < m
    l = 0
    for i in range(idx):
        l += get_split_sz(n, m, i)
    r = l + get_split_sz(n, m, idx)
    return slice(l, r)

@numba.njit
def get_rand_idx_numba(l, r, cnt):
    if l<r:
        return np.random.choice(np.arange(l, r), cnt, replace=False)
    return None

def parse_dt_mode_to_src_tgt_dim(dt_mode:str, state_dim:int, action_dim:int):
    src_dim_mapping = {
        's': state_dim,
        'sa': state_dim + action_dim,
        'as': state_dim + action_dim,
    }
    tgt_dim_mapping = {
        'a': action_dim,
        'sa': state_dim + action_dim,
        'as': state_dim + action_dim,
    }
    return src_dim_mapping[dt_mode.split('_')[0]], tgt_dim_mapping[dt_mode.split('_')[1]]

def get_py_virtual_display(size):
    backends = ['xvfb', 'xephyr', 'xvnc']
    display = None
    for backend in backends:
        try:
            display = SmartDisplay(size=size, backend=backend)
            break
        except:
            display = None
    if display is None:
        raise ValueError('Error: no backend is supportted!')
    return display

def convert_tensor_dict_to_tensor_list(x:dict):
    if isinstance(x, torch.Tensor):
        return [x]
    if not isinstance(x, dict):
        return []
    res = []
    for k, v in x.items():
        res += convert_tensor_dict_to_tensor_list(v)
    return res

def merge_tensor_dict_to_single_tensor(x:dict) -> torch.Tensor:
    y = convert_tensor_dict_to_tensor_list(x)
    return torch.cat(y, dim=-1)

def tensor_copy_or_not(x:torch.Tensor, copy=False) -> torch.Tensor:
    return x.detach().clone() if copy else x

def merge_tensor_dict_to_single_tensor_fixed(x:dict, env, task:str, dt_mode="s_a") -> torch.Tensor:
    '''
        state space of tasks:
            1.unitree_general
                projected_gravity: 3
                dof_vel: 12
                dof_pos: sin 12
                dof_pos: cos 12
                base_ang_vel: 3
                actions: 12
                commands: 16
                # contact_forces: 51 # actor can not get, which is critic info
        dt_mode: s_a, as_a. default: s_a. default dt_mode is used for data collection...
    '''
    if task == 'unitree_general':
        actor_obs_tensor = x['actor']['nn'][..., :env.cfg.env.actor_num_obs]
        if dt_mode == "s_a":
            return torch.cat([actor_obs_tensor[..., :42], actor_obs_tensor[..., -16:]], dim=-1)
        elif dt_mode == "as_a":
            return torch.cat([actor_obs_tensor[..., 42:54], 
                              actor_obs_tensor[..., :42], actor_obs_tensor[..., -16:]], dim=-1)
    if task in ["halfcheetah", "hopper", "walker2d", "ant"]:
        pass
        # action? init action?
    return x['actor']['nn'][..., :env.cfg.env.actor_num_obs]
    # return torch.cat([x['actor']['nn'][..., :env.cfg.env.actor_num_obs], 
    #                   x['critic']['nn'][..., :env.cfg.env.critic_num_obs]], dim=-1)
    # return torch.cat([x['actor']['nn'], x['actor']['ontology_sense']['nn'], 
    #                   x['critic']['nn']], dim=-1)

# 目前不需要
# def split_single_tensor_to_tensor_dict_fixed(x:torch.Tensor, copy=False) -> dict:
#     return {
#         'actor': {
#             'nn': tensor_copy_or_not(x[:, :512], copy),
#             'ontology_sense': {
#                 'nn': tensor_copy_or_not(x[:, 512:-512], copy),
#             },
#         },
#         'critic': {
#             'nn': tensor_copy_or_not(x[:, -512:], copy),
#         },
#     }

def encode_arg_names_into_arg_list(local_vars:dict, global_vars:dict, processors:int, arg_names:list):
    arg_list = []
    for arg_name in arg_names:
        arg_val = local_vars[arg_name] if arg_name in local_vars else global_vars[arg_name]
        arg_list.append([arg_val for _ in range(processors)])
    arg_list.append([i for i in range(processors)])
    arg_list.append([arg_names for _ in range(processors)])
    return list(zip(*arg_list))

def parse_dt_policy_dir(dt_policy_dir:str):
    '''
        @return: (dt_policy_path[str], config[dict])
    '''
    assert os.path.exists(dt_policy_dir), f'dt_policy_dir: {dt_policy_dir} does not exist.'
    dt_policy_name, dt_config_name = None, None
    for file_name in os.listdir(dt_policy_dir):
        if '.json' in file_name:
            dt_config_name = file_name
        elif '.pth' in file_name:
            dt_policy_name = file_name
    assert dt_policy_name is not None and dt_config_name is not None, f'dt_policy_dir: {dt_policy_dir}, dt_policy_name: {dt_policy_name}'
    with open(join(dt_policy_dir, dt_config_name), 'r') as f:
        config_c = f.read()
    config = json.loads(config_c)

    return join(dt_policy_dir, dt_policy_name), config

class MultiAgentObsBuffer:
    def __init__(self, num_envs, seq_len, max_episode_length, state_dim, action_dim, tgt_dim, device, dtype) -> None:
        self.num_envs = num_envs
        self.seq_len = seq_len
        self.max_episode_length = max_episode_length
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tgt_dim = tgt_dim
        self.device = device
        self.dtype = dtype
        self.obs_buf = torch.zeros((num_envs, max_episode_length + 100, state_dim), dtype=dtype, device=device)
        self.offset = torch.arange(seq_len).reshape(1, -1).repeat_interleave(num_envs, dim=0).to(device)
        self.buf_idx = torch.zeros((num_envs, ), dtype=torch.int64, device=device)
        return
    
    def update(self, obs:torch.Tensor):
        '''
            @return attn_mask: (num_envs, seq_len)
            @return tgt_selector: (num_envs, 1, tgt_dim)
        '''
        obs_buf = self.obs_buf
        buf_idx = self.buf_idx
        seq_len = self.seq_len
        state_dim = self.state_dim
        action_dim = self.action_dim
        tgt_dim = self.tgt_dim
        offset = self.offset
        num_envs = self.num_envs
        device = self.device
        dtype = self.dtype
        assert state_dim == obs.shape[-1], f"obs_shape: {obs.shape}"
        
        obs = obs.type(dtype)
        obs_buf.scatter_(
            1,
            buf_idx.reshape(-1, 1, 1).repeat_interleave(state_dim, dim=2),
            obs.detach().unsqueeze(1),
        )
        buf_idx += 1
        
        start_pos = torch.where(buf_idx >= seq_len, buf_idx - seq_len, 0).reshape(-1, 1)\
            .repeat_interleave(seq_len, dim=1).to(device)
        state_mask = (start_pos + offset).reshape(num_envs, seq_len, 1)\
            .repeat_interleave(state_dim, dim=2).to(device)
        states = torch.gather(obs_buf, 1, state_mask)
        
        idx = buf_idx.detach().clone().reshape(-1, 1).repeat_interleave(seq_len, dim=1)

        attn_mask = torch.arange(seq_len).reshape(1, -1).repeat_interleave(num_envs, dim=0).to(device)
        attn_mask = torch.where(attn_mask < idx, True, False).to(device)
        
        tgt_selector = torch.where(buf_idx >= seq_len, seq_len - 1, buf_idx - 1).to(device)
        tgt_selector = tgt_selector.reshape(-1, 1, 1).repeat_interleave(tgt_dim, dim=2).to(device)
        
        return states, attn_mask, tgt_selector
    
    def post_update(self, done):
        buf_idx = self.buf_idx
        new_ids = torch.nonzero(done > 0).reshape(-1)
        buf_idx[new_ids] = 0
        return

class MultiAgentActionBuffer:
    def __init__(self, num_envs, seq_len, max_episode_length, action_dim, device, dtype) -> None:
        self.num_envs = num_envs
        self.seq_len = seq_len
        self.max_episode_length = max_episode_length
        self.action_dim = action_dim
        self.device = device
        self.dtype = dtype
        self.action_buf = torch.zeros((num_envs, max_episode_length + 100, action_dim), dtype=dtype, device=device)
        self.offset = torch.arange(seq_len).reshape(1, -1).repeat_interleave(num_envs, dim=0).to(device)
        self.buf_idx = torch.zeros((num_envs, ), dtype=torch.int64, device=device)
        return
    
    def get_action_las_t(self):
        action_buf = self.action_buf
        buf_idx = self.buf_idx # (num_envs, )
        seq_len = self.seq_len
        action_dim = self.action_dim
        offset = self.offset
        num_envs = self.num_envs
        device = self.device
        
        start_pos = torch.where(buf_idx >= seq_len, buf_idx - seq_len, 0).reshape(-1, 1)\
            .repeat_interleave(seq_len, dim=1).to(device) # (num_envs, seq_len)
        state_mask = (start_pos + offset).reshape(num_envs, seq_len, 1)\
            .repeat_interleave(action_dim, dim=2).to(device)
        actions = torch.gather(action_buf, 1, state_mask) # (num_envs, seq_len, action_dim)

        shift_idx = (buf_idx < seq_len).nonzero(as_tuple=True)[0]
        actions[shift_idx, 1:, :] = actions[shift_idx, :-1, :] # TODO: 有点问题，steps<=seq_len的env 和 steps>seq_len的env 是不一样的
        actions[shift_idx, 0, :] = 0. # TODO: episode 一开始的action是设为0，还是default dof position
        return actions

    def update(self, action:torch.Tensor):
        action_buf = self.action_buf
        buf_idx = self.buf_idx # (num_envs, )
        action_dim = self.action_dim
        dtype = self.dtype
        assert action_dim == action.shape[-1], f"action_shape: {action.shape}"
        
        action = action.type(dtype)

        # print(f'action_shape: {action.shape}, action_dtype: {action.dtype}')

        # action_buf shape: (num_envs, seq_len, action_dim)
        action_buf.scatter_(
            1,
            buf_idx.reshape(-1, 1, 1).repeat_interleave(action_dim, dim=2),
            action.detach().unsqueeze(1),
        )
        buf_idx += 1
        return
    
    def post_update(self, done):
        buf_idx = self.buf_idx
        new_ids = torch.nonzero(done > 0).reshape(-1)
        buf_idx[new_ids] = 0
        return

@numba.njit
def convert_episodes_to_samples_append_state_action(
    src_res: np.ndarray, tgt_res: np.ndarray, sample_num_idx: int, 
    state: np.ndarray, action: np.ndarray, eplen: int, sample_pos: int,
    src_mode: str, tgt_mode: str, state_dim: int, action_dim: int,
    seq_len: int
):
    if src_mode == 's':
        src_res[sample_num_idx, :, :] = state[sample_pos:sample_pos + seq_len, :].copy()
    elif src_mode == 'sa':
        src_res[sample_num_idx, :, :state_dim] = state[sample_pos:sample_pos + seq_len, :].copy()
        if sample_pos == 0:
            src_res[sample_num_idx, 1:, -action_dim:] = action[:seq_len - 1, :].copy()
            # src_res[sample_num_idx, 0, -action_dim:] = action[0, :].copy() # TODO
        else:
            src_res[sample_num_idx, :, -action_dim:] = action[
                                                        sample_pos - 1:sample_pos + seq_len - 1,
                                                        :].copy()
    elif src_mode == 'as':
        src_res[sample_num_idx, :, -state_dim:] = state[sample_pos:sample_pos + seq_len, :].copy()
        if sample_pos == 0:
            src_res[sample_num_idx, 1:, :action_dim] = action[:seq_len - 1, :].copy()
            # src_res[sample_num_idx, 0, -action_dim:] = action[0, :].copy() # TODO
        else:
            src_res[sample_num_idx, :, :action_dim] = action[sample_pos - 1:sample_pos + seq_len - 1,
                                                        :].copy()
    else:
        print('dt_mode_src:', src_mode, 'not in [s, sa, as]')

    if tgt_mode == 'a':
        tgt_res[sample_num_idx, :, :] = action[sample_pos:sample_pos + seq_len, :].copy()
    elif tgt_mode == 'sa':
        tgt_res[sample_num_idx, :, -action_dim:] = action[sample_pos:sample_pos + seq_len, :].copy()
        if sample_pos + seq_len == eplen:
            tgt_res[sample_num_idx, :-1, :state_dim] = state[sample_pos + 1:sample_pos + seq_len,
                                                        :].copy()
            tgt_res[sample_num_idx, -1, :state_dim] = state[sample_pos + seq_len - 1,
                                                        :].copy()  # TODO
        else:
            tgt_res[sample_num_idx, :, :state_dim] = state[sample_pos + 1:sample_pos + seq_len + 1,
                                                        :].copy()
    elif tgt_mode == 'as':
        tgt_res[sample_num_idx, :, :action_dim] = action[sample_pos:sample_pos + seq_len, :].copy()
        if sample_pos + seq_len == eplen:
            tgt_res[sample_num_idx, :-1, -state_dim:] = state[sample_pos + 1:sample_pos + seq_len,
                                                        :].copy()
            tgt_res[sample_num_idx, -1, -state_dim:] = state[sample_pos + seq_len - 1,
                                                        :].copy()  # TODO
        else:
            tgt_res[sample_num_idx, :, -state_dim:] = state[sample_pos + 1:sample_pos + seq_len + 1,
                                                        :].copy()
    else:
        print('dt_mode_tgt:', tgt_mode, 'not in [a, sa, as]')

# @numba.njit
def convert_episodes_to_samples_nonmdp_numba(
    state_eps:np.ndarray, action_eps, eplen_eps, seq_len,
    now_sample_cnt, total_sample_tgt, state_dim, action_dim,
    src_dim, tgt_dim, src_mode:str, tgt_mode:str
):
    '''
        :param state_eps: [] # states (-1, max(ep_len)+10, state_dim,)
        :param action_eps: [] # actions (-1, max(ep_len)+10, action_dim,)
        :return state_res: [] (sample_num, seq_len, state_dim, )
        :return action_res: [] (sample_num, seq_len, action_dim, )
    '''
    total_sample_num = 0
    for _, (state, action, eplen) in enumerate(zip(state_eps, action_eps, eplen_eps)):
        if eplen >= seq_len:
            total_sample_num += 1
        if eplen > seq_len:
            total_sample_num += (eplen - ((eplen - 1) % seq_len + 1)) // seq_len
    total_sample_num = min(total_sample_num, total_sample_tgt - now_sample_cnt)
    print("total_sample_num * seq_len", total_sample_num, seq_len)
    
    src_res = np.zeros((total_sample_num, seq_len, src_dim), dtype=np.float32)
    tgt_res = np.zeros((total_sample_num, seq_len, tgt_dim), dtype=np.float32)

    sample_num_idx = 0
    for _, (state, action, eplen) in enumerate(zip(state_eps, action_eps, eplen_eps)):
        if eplen >= seq_len:
            if _ < 1:
                print("episode steps: start end", 0, seq_len-1)
            convert_episodes_to_samples_append_state_action(
                src_res, tgt_res, sample_num_idx, state, action, eplen, 0,
                src_mode, tgt_mode, state_dim, action_dim, seq_len)
            sample_num_idx += 1
            if sample_num_idx >= total_sample_num: break
        if eplen > seq_len:
            # start_rand_pos = eplen % seq_len
            # start_pos = get_rand_idx_numba(start_rand_pos, start_rand_pos + seq_len, 1)[0]
            start_pos = (eplen - 1) % seq_len + 1
            if _ < 1:
                print("episode steps: start end", start_pos, eplen-1)
            for sample_pos in range(start_pos, eplen, seq_len):
                convert_episodes_to_samples_append_state_action(
                    src_res, tgt_res, sample_num_idx, state, action, eplen, sample_pos,
                    src_mode, tgt_mode, state_dim, action_dim, seq_len)
                sample_num_idx += 1
                if sample_num_idx >= total_sample_num: break
            if sample_num_idx >= total_sample_num: break

    if sample_num_idx != total_sample_num:
        print('in numba ValueError: sample_num_idx:', sample_num_idx, ' do not equal to total_sample_num:',
              total_sample_num)
        # raise ValueError

    return src_res, tgt_res

@numba.njit
def convert_episodes_to_samples_mdp_numba(state_eps:np.ndarray, action_eps, eplen_eps,
                                      now_sample_cnt, total_sample_tgt, state_dim, action_dim,
                                      src_dim, tgt_dim, src_mode:str, tgt_mode:str):
    this_sample_num = 0
    # state: (ep_len, state_dim), action: (ep_len, action_dim), eplen: int
    for _, (state, action, eplen) in enumerate(zip(state_eps, action_eps, eplen_eps)):
        if eplen >= 10: # filter super short episodes
            this_sample_num += eplen
    this_sample_num = min(this_sample_num, total_sample_tgt - now_sample_cnt)

    src_res = np.zeros((this_sample_num, src_dim, 1), dtype=np.float32)
    tgt_res = np.zeros((this_sample_num, tgt_dim, 1), dtype=np.float32)

    sample_idx = 0
    # state: (ep_len, state_dim), action: (ep_len, action_dim), eplen: int
    for _, (state, action, eplen) in enumerate(zip(state_eps, action_eps, eplen_eps)):
        if eplen >= 10: # filter super short episodes
            ep_sample_num = min(eplen, this_sample_num - sample_idx)
            if src_mode == "s" or src_mode == "sa":
                src_res[sample_idx: sample_idx + ep_sample_num, :state_dim, 0] = \
                    state[:ep_sample_num, :].copy()
            elif src_mode == "as":
                src_res[sample_idx: sample_idx + ep_sample_num, -state_dim:, 0] = \
                    state[:ep_sample_num, :].copy()

            if src_mode == "sa":
                src_res[sample_idx + 1: sample_idx + ep_sample_num, -action_dim:, 0] = \
                    action[:ep_sample_num - 1, :].copy()
            elif src_mode == "as":
                src_res[sample_idx + 1: sample_idx + ep_sample_num, :action_dim, 0] = \
                    action[:ep_sample_num - 1, :].copy()

            if tgt_mode == 'a':
                tgt_res[sample_idx: sample_idx + ep_sample_num, :action_dim, 0] = \
                    action[:ep_sample_num, :].copy()

            sample_idx += ep_sample_num
            if sample_idx >= this_sample_num: break

    if sample_idx != this_sample_num:
        print('sample_idx:', sample_idx, ' do not equal to this_sample_num:', this_sample_num)
        print('Numba function convert_episodes_to_samples_mdp_numba ValueError')
        raise ValueError

    return src_res, tgt_res

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total_Parameters: {total_num:,}, Trainable_Parameters: {trainable_num:,}")
    return {'Total': total_num, 'Trainable': trainable_num}

def get_parameter_number_no_output(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total_Parameters: {total_num:,}, Trainable_Parameters: {trainable_num:,}")
    return {'Total': total_num, 'Trainable': trainable_num}

def calc_new_env_rew(infos:dict):
    rews_tensor_list = []
    for rew_name in infos['episode']:
        if 'rew_' in rew_name:
            rews_tensor_list.append(infos['episode'][rew_name])
    return sum(rews_tensor_list)

def convert_tensor_to_shape(x:Union[tuple, list, dict]):
    '''
    @x: a dict or tuple or list contains torch.Tensor or np.ndarray.
    '''
    if isinstance(x, (tuple, list)):
        res = []
        for y in x:
            res.append(convert_tensor_to_shape(y))
        if isinstance(x, tuple):
            res = tuple(res)
        return res
    elif isinstance(x, dict):
        res = {}
        for k, v in x.items():
            res[k] = convert_tensor_to_shape(v)
        return res
    elif isinstance(x, (np.ndarray, torch.Tensor)):
        return x.shape
    else:
        return x

def normal_samples_to_bz_multiplier(sample_cnt: int, batch_size: int):
    return sample_cnt // batch_size * batch_size

def clear_components(dir: str):
    shutil.rmtree(dir, ignore_errors=True)
    os.makedirs(dir, exist_ok=True)


# episode save data & train
def save_np_mmap(x: np.ndarray, data_dir: str, file_name: str, file_format: str):
    data_path = join(data_dir, f"{file_name}_{'_'.join(map(str, x.shape))}.{file_format}")
    fp = np.memmap(data_path, dtype=x.dtype, mode='w+', shape=x.shape)
    fp[:] = x[:]
    fp.flush()
    del fp

def match_pattern_in_list(x:list, y:str):
    x_pattern = re.compile(y)
    matched_xs = list(filter(lambda z: bool(x_pattern.match(z)), x))
    return matched_xs

def read_np_mmap(data_dir: str, file_name: str, file_format: str, dtype, np_memmap_mode: str):
    data_names = list(os.listdir(data_dir))
    pattern = rf"{file_name}_(.*).{file_format}"
    matched_data_names = match_pattern_in_list(data_names, pattern)
    assert len(matched_data_names) == 1, f"{matched_data_names}, {data_names}, {pattern}"
    matched_data_name = matched_data_names[0]
    data_path = join(data_dir, matched_data_name)
    shape_str: str = matched_data_name[len(file_name)+1:-(len(file_format)+1)]
    x_shape = tuple(map(int, shape_str.split('_')))
    fp = np.memmap(data_path, dtype=dtype, mode=np_memmap_mode, shape=x_shape)
    return fp

def split_np2tensorList(x: np.ndarray, meta: np.ndarray, max_episode_length: int = -1,
                        return_type: int = 'torch'):
    # split the whole x into several samples according to meta
    # res: list of samples
    # cur_eplen: current index of the whole x
    res, cur_eplen = [], 0
    for eplen in meta:
        sample_eplen = eplen
        if max_episode_length > 0 and sample_eplen > max_episode_length:
            sample_eplen = max_episode_length
        
        if return_type == 'torch':
            res.append(torch.from_numpy(x[cur_eplen: cur_eplen + sample_eplen, :]))
        elif return_type == 'numpy':
            res.append(x[cur_eplen: cur_eplen + sample_eplen, :])
        else:
            raise NotImplementedError
        
        cur_eplen += eplen
    return res

# sync multi-gpu
def sync_mgpu(x: Union[int, float, torch.Tensor], device, opt: str):
    is_tensor = isinstance(x, torch.Tensor)
    if not is_tensor:
        x = torch.tensor([x], device=device)
    else:
        x = x.to(device)
    opt2torchOp = {
        'min': torch.distributed.ReduceOp.MIN,
        'max': torch.distributed.ReduceOp.MAX,
        'avg': torch.distributed.ReduceOp.AVG,
    }
    torch.distributed.all_reduce(x, op=opt2torchOp[opt])
    if is_tensor:
        return x
    return x.cpu().item()