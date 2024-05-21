from typing import List, Dict, Union, Optional
import os
from os.path import join
import bisect
import aligo
import re
import operator
from functools import reduce, partial
import lzma
import pickle
import random
import statistics

import numpy as np
np.set_printoptions(suppress=True, formatter={'float_kind': '{:16.3f}'.format}, linewidth=1000_000_000,
                    threshold=np.inf)

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler

from FCNet.DT.utils.common import chunk_into_slices, normal_samples_to_bz_multiplier, \
    read_np_mmap, split_np2tensorList, match_pattern_in_list, sync_mgpu, \
    convert_episodes_to_samples_nonmdp_numba, parse_dt_mode_to_src_tgt_dim
from FCNet.DT.utils.type_utils import FLOAT_TYPE, INT_TYPE, CHUNK_LOAD_MODES, EPISODE_LOAD_MODES
from FCNet.DT.utils.name_utils import EPISODE_LOAD_DATA_RE

# multi task dataset, but dataloader require same dim size
class ChunkDataset(Dataset):
    def __init__(self, srcs:list, tgts:list):
        super().__init__()
        self.srcs, self.tgts = srcs, tgts
        self.lengths = []
        for src in srcs:
            self.lengths.append(src.shape[0])
        self._total_length = sum(self.lengths)
        self.indexs = [0 for _ in range(len(self.lengths))]
        for i in range(1, len(self.indexs)):
            self.indexs[i] = self.indexs[i-1] + self.lengths[i-1]
        return
    
    @property
    def total_length(self):
        return self._total_length
    
    @total_length.setter
    def total_length(self, value):
        self._total_length = value
    
    def __getitem__(self, index):
        belong_data_index = bisect.bisect_right(self.indexs, index) - 1
        real_index = index - self.indexs[belong_data_index]
        return self.srcs[belong_data_index][real_index], self.tgts[belong_data_index][real_index]

    def __len__(self):
        return self._total_length
    


# tmporarily one task
class EpisodeDataset(Dataset):
    def __init__(self, src_episodes: List[torch.Tensor], tgt_episodes: List[torch.Tensor]):
        super().__init__()
        self.src_episodes, self.tgt_episodes = src_episodes, tgt_episodes
        self.max_episode_length = sum(map(lambda x: x.size(0), self.src_episodes))
        return
    
    def __getitem__(self, index):
        return self.src_episodes[index], self.tgt_episodes[index]

    def __len__(self):
        return len(self.src_episodes)


class ChunkCollator(object):
    '''
    customed pytorch dataloader collate_fn
    '''
    def __init__(self, src_dim, tgt_dim):
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
    def __call__(self, batch:list):
        "batch: list[tuple[src, tgt]]"
        batch_0_src, batch_0_tgt = batch[0]
        device, dtype, bz, seq_len = batch_0_src.device, \
            batch_0_src.dtype, len(batch), batch_0_src.shape[0]
        src = torch.zeros((bz, seq_len, self.src_dim), dtype=dtype, device=device)
        tgt = torch.zeros((bz, seq_len, self.tgt_dim), dtype=dtype, device=device)
        for i, (s, t) in enumerate(batch):
            src[i, :, :s.shape[-1]] = s[:, :]
            tgt[i, :, :t.shape[-1]] = t[:, :]
        return src, tgt

# class EpisodeCollator(object):
#     def __init__(self, src_dim, tgt_dim):
#         self.src_dim = src_dim
#         self.tgt_dim = tgt_dim
#         return
    
#     def __call__(self, batch:list):
#         "batch: list[tuple[src, tgt]]"
#         ele = batch[0]
#         device = ele[0].device
#         dtype = ele[0].dtype
#         bz = len(batch)
#         max_eplen = max(batch, key=lambda x: x[0].shape[-2])[0].shape[-2]
#         src = torch.zeros((bz, max_eplen, self.src_dim), dtype=dtype, device=device)
#         tgt = torch.zeros((bz, max_eplen, self.tgt_dim), dtype=dtype, device=device)
#         for i, (s, t) in enumerate(batch):
#             src[i, :s.shape[-2], :s.shape[-1]] = s[:, :]
#             tgt[i, :t.shape[-2], :t.shape[-1]] = t[:, :]
#         return src, tgt

# class Dataset_mk(Dataset):
#     def __init__(self, src:torch.Tensor, tgt:torch.Tensor):
#         super().__init__()
#         self.src, self.tgt = src, tgt
#         return
    
#     def __getitem__(self, index):
#         return self.src[index], self.tgt[index]

#     def __len__(self):
#         return self.src.shape[0]


class EpisodeLoader:
    def __init__(self, dataset: EpisodeDataset, batch_size: int, shuffle: bool = True,
                 device = None, distributed: bool = False) -> None:
        if len(dataset) < batch_size:
            batch_size = len(dataset) // 2 # 至少两个epoch
            print(f"dataset too small, reset batch_size to the size of dataset: {batch_size}.")
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.indexs = [i for i in range(len(self.dataset))]
        self.clip_eplens = [random.randint(1, dataset[i][0].size(0)) for i in range(batch_size)]
        
        self.max_episode_length = max(map(lambda x: x[0].size(0), dataset))
        if distributed:
            assert device is not None, device
            self.max_episode_length = sync_mgpu(self.max_episode_length, device, 'max')
        
        self.src_dim = dataset[0][0].size(-1)
        self.tgt_dim = dataset[0][1].size(-1)
        
        self.src_buf = torch.zeros((self.batch_size, self.max_episode_length, self.src_dim), dtype=torch.float32)
        self.tgt_buf = torch.zeros((self.batch_size, self.max_episode_length, self.tgt_dim), dtype=torch.float32)
        self.buf_len = torch.zeros((self.batch_size,), dtype=torch.int64)
        self.buf_idx = torch.zeros((self.batch_size,), dtype=torch.int64)
        
        self.iter_max = None
        
        self.reshuffle_dataset()
        # self.loader_length = int(np.ceil(sum(map(lambda x: x[0].size(0), dataset)) / batch_size)) # roughly estimate loader_length
    
    @property
    def real_length(self):
        if self.iter_max is not None:
            return min(self.iter_max, self.loader_length)
        return self.loader_length
    
    def __len__(self):
        ''' Rough estimate, not accurate '''
        return self.loader_length
    
    def __iter__(self):
        self.iter_idx = 0 # index of dataset
        self.iter_times = 0 # times of __next__ invoked
        
        ''' reset buf to 0 '''
        self.src_buf[:], self.tgt_buf[:], self.buf_len[:], self.buf_idx[:] = 0, 0, 0, 0
        return self
    
    def __next__(self):
        '''
        @return src_single_step: (batch_size, src_dim)
        @return tgt_single_step: (batch_size, tgt_dim)
        @return need_new_mask: (batch_size) # indicate which env is at the first step of this episode
        '''
        self.iter_times += 1
        if self.iter_max is not None and self.iter_times > self.iter_max:
            raise StopIteration
        
        need_new_mask = self.buf_idx >= self.buf_len
        need_new_ids = torch.nonzero(need_new_mask).reshape(-1)
        
        self.src_buf[need_new_ids], self.tgt_buf[need_new_ids] = 0, 0
        for need_new_id in need_new_ids:
            if self.iter_idx >= len(self.dataset):
                break
            src_episode, tgt_episode = self.dataset[self.indexs[self.iter_idx]]
            eplen = src_episode.size(0)
            if self.iter_idx < self.batch_size:
                eplen = self.clip_eplens[self.iter_idx]
            self.src_buf[need_new_id, :eplen, :] = src_episode[:eplen, :]
            self.tgt_buf[need_new_id, :eplen, :] = tgt_episode[:eplen, :]
            self.buf_len[need_new_id] = eplen
            self.buf_idx[need_new_id] = 0
            self.iter_idx += 1
        
        invalid_ids = torch.nonzero(self.buf_idx >= self.buf_len).reshape(-1)
        if invalid_ids.numel() > 0:
            raise StopIteration
        # reduce (8, 12, 2) to (8, 2) by a index (8,)
        tmp_select_tensor = torch.arange(self.batch_size)
        src_single_step, tgt_single_step = self.src_buf[tmp_select_tensor, self.buf_idx], \
            self.tgt_buf[tmp_select_tensor, self.buf_idx]
        self.buf_idx += 1
        return src_single_step, tgt_single_step, need_new_mask
    
    def _calc_precise_len(self):
        tmp_iter_max = self.iter_max
        self.iter_max = None
        
        self.loader_length = 0
        for _ in self:
            self.loader_length += 1
        
        self.iter_max = tmp_iter_max
    
    def reshuffle_dataset(self):
        if self.shuffle:
            random.shuffle(self.indexs)
            for i in range(self.batch_size):
                self.clip_eplens[i] = random.randint(1, self.dataset[self.indexs[i]][0].size(0))
        self._calc_precise_len()
    
    def set_iter_max(self, iter_max: int):
        self.iter_max = iter_max


def check_data_validity(
    aligo_name: str, aligo_data_root_dir: str, data_root_dir: str, 
    tasks: list, aligo_enable: bool, data_mode: str, task_config: dict
) -> List[str]:
    if data_mode == 'mdp':
        raise NotImplementedError
        return
    
    data_files = os.listdir(data_root_dir)
    data_files_in_tasks = list()
    for task in tasks:
        # match files use re
        assert task in task_config, f'{task} not in task_config'
        player_level = task_config[task]['player_level']
        pattern = eval(EPISODE_LOAD_DATA_RE) # use player_level
        print("data_root_dir, pattern", data_root_dir, pattern)
        matched_data_files = match_pattern_in_list(data_files, pattern)
        
        # check matched file cnt
        if len(matched_data_files) == 0:
            if aligo_enable:
                pass
                # TODO: add reward limit flexibly support
                # print(f'aligo downloading {data_file}...')
                # # download folder
                # from aligo import Aligo
                # ali = Aligo(name=aligo_name)
                # remote_folder = ali.get_folder_by_path(join(aligo_data_root_dir, data_file))
                # ali.download_folder(remote_folder.file_id, local_folder=data_root_dir)
                # assert os.path.exists(join(data_root_dir, data_file)), f'{data_file} not in {os.listdir(data_root_dir)}'
                # print(f'aligo downloading {data_file} End.')
            else:
                raise ValueError('aligo banned, but data does not exist!')
        elif len(matched_data_files) > 1:
            raise ValueError(f'{len(matched_data_files)} datas: {matched_data_files} \
meet the condition, check it!')
        
        # append good files
        data_files_in_tasks.append(matched_data_files[0])
    return data_files_in_tasks

def make_dataloader(
    dataset, my_collator, batch_size, shuffle_dataset=True, random_seed=42,
):
    indices = list(range(len(dataset)))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    # print('Use DistributedSampler!')
    # sampler = DistributedSampler(indices)
    print('Making Dataloader Use SubsetRandomSampler!')
    sampler = SubsetRandomSampler(indices)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=my_collator,
    )
    return data_loader

def loadData_from_disk_chunkMode(local_vars: dict, return_type='torch'):
    '''
    load data from disk if the data is in chunk mode
    @return_type: [numpy, torch]
    @dataset_info_buf: contain info of the dataset
    @data_read_from: e.g. ../../../../share/FCNet/data
    @data_file: unitree_general_60_s_a_chunk_r-100
    @max_sample_number: only avaliable in chunk mode
    @return src: torch.Tesnor
    @return tgt: torch.Tesnor
    '''
    print("loading data from disk in chunk mode.")

    # check return_type validation
    assert return_type in ['numpy', 'torch'], return_type
    
    # load vars from locals()
    dataset_info_buf = local_vars['dataset_info_buf']
    data_read_from = local_vars['data_read_from']
    data_file = local_vars['data_file']
    data_scale = local_vars['data_scale']
    batch_size = local_vars['batch_size']
    world_size = local_vars['world_size']
    train_ratio = local_vars['train_ratio']
    tasks = local_vars['tasks']
    local_rank = local_vars['local_rank']
    
    # concat data path
    data_dir = join(data_read_from, data_file)
    data_list = list(os.listdir(data_dir)) # e.g. [src_1000000_60_58.dat, tgt_1000000_60_12.dat]
    
    # parse data's name and shape from *.dat file name
    def parse_name_shape(data_type: str):
        '''
        @data_type: e.g. [src, tgt]
        '''
        for data_name_elem in data_list:
            if data_name_elem.split('.')[0].split('_')[0] == data_type:
                return data_name_elem, tuple(map(int, data_name_elem.split('.')[0].split('_')[1:]))
        return None, None
    # e.g. src_1000000_60_58.dat, (1000000, 60, 58), ...
    src_data_name, src_data_shape, tgt_data_name, tgt_data_shape = \
        reduce(operator.add, list(map(parse_name_shape, ['src', 'tgt'])))
    assert src_data_name is not None and tgt_data_name is not None, \
        f'src_data_name: {src_data_name}, tgt_data_name: {tgt_data_name}'

    # update dataset infomation
    dataset_info_buf[data_file] = [src_data_name, tgt_data_name]
    
    # make memmap fp pointer fron disk
    src_data_path, tgt_data_path = list(map(lambda x: join(data_dir, x), [src_data_name, tgt_data_name]))
    src_fp, tgt_fp = list(map(lambda x: np.memmap(x[0], dtype=np.float32, mode='r', shape=x[1]),
                            [(src_data_path, src_data_shape), (tgt_data_path, tgt_data_shape)]))
    src_data_shape, tgt_data_shape = list(map(list, [src_data_shape, tgt_data_shape]))
    if data_scale:
        max_sample_number = min(max_sample_number,
                                int(batch_size * 1.1 * world_size / (1.0 - train_ratio) / len(tasks)))
    src_data_shape[0], tgt_data_shape[0] = list(map(lambda x: min(x, max_sample_number),
                                                    [src_data_shape[0], tgt_data_shape[0]]))
    src_data_shape[0], tgt_data_shape[0] = list(map(lambda x: normal_samples_to_bz_multiplier(x, batch_size),
                                                    [src_data_shape[0], tgt_data_shape[0]]))
    src_data_shape, tgt_data_shape = list(map(tuple, [src_data_shape, tgt_data_shape]))

    src_slice, tgt_slice = list(map(lambda x: chunk_into_slices(x, world_size)[local_rank],
                                    [src_data_shape[0], tgt_data_shape[0]]))
    print(f"src_slice: {src_slice}, tgt_slice: {tgt_slice}")
    
    # make numpy[src, tgt] memory space and load data from disk to memory
    this_rank_src_shape = (int(src_slice.stop - src_slice.start), src_data_shape[1], 
                            src_data_shape[2])
    this_rank_tgt_shape = (int(tgt_slice.stop - tgt_slice.start), tgt_data_shape[1], 
                            tgt_data_shape[2])
    src_np = np.zeros(this_rank_src_shape, dtype=np.float32)
    tgt_np = np.zeros(this_rank_tgt_shape, dtype=np.float32)
    src_np[:, :, :] = src_fp[src_slice, :, :]
    tgt_np[:, :, :] = tgt_fp[tgt_slice, :, :]
    print(f'Local_rank: {local_rank}, reading from disk Done!')
    
    if return_type == 'torch':
        # convert numpy.ndarray to torch.Tensor, sharing memory
        src = torch.from_numpy(src_np)
        tgt = torch.from_numpy(tgt_np)
        print(f'Local_rank: {local_rank}, torch from_numpy Done!')
    elif return_type == 'numpy':
        src, tgt = src_np, tgt_np
    
    return src, tgt

def print_info_buf_exec(print_info_buf: dict):
    '''
    print out buffer dict
    @print_info_buf: dict(
        foo: eval(foo),
        ...
    )
    '''
    print_info_str_list = list()
    for k, v in print_info_buf.items():
        if isinstance(v, FLOAT_TYPE):
            print_info_str_list.append(f'{k}: {v:.4f}')
        elif isinstance(v, INT_TYPE):
            print_info_str_list.append(f'{k}: {v}')
        else:
            print_info_str_list.append(f'{k}: {v}')
    print_info_str = ', '.join(print_info_str_list)
    print(print_info_str)

def split_train_test_1d(whole_data: Union[np.ndarray, torch.Tensor], train_ratio: float):
    '''
    split the whole data alongside 1-dimension
    '''
    whole_len = whole_data.shape[0]
    train_len = int(whole_len * train_ratio)
    
    train_data = whole_data[:train_len, ...]
    test_data = whole_data[train_len:, ...]
    
    return train_data, test_data

def loadData_from_disk_episodeMode(local_vars: dict, return_type='torch'):
    '''
    load data from disk if the data is in episode mode
    '''
    print("loading data from disk in episode mode.")
    
    assert return_type in ['numpy', 'torch'], return_type
    
    # load vars from locals()
    print_info_buf = local_vars['print_info_buf']
    data_read_from = local_vars['data_read_from']
    data_file = local_vars['data_file']
    local_rank = local_vars['local_rank']
    world_size = local_vars['world_size']
    max_episode_length = local_vars['max_episode_length']
    # three limitations
    # max_sample_number = local_vars['max_sample_number'] # only avaliable in chunk mode
    max_episode_length = local_vars['max_episode_length']
    max_train_steps = local_vars['max_train_steps']
    
    # get this local rank's max_train_steps
    if max_train_steps is not None:
        max_train_steps //= world_size
    
    # concat data paths
    data_dir = join(data_read_from, data_file)
    data_list = list(os.listdir(data_dir))

    # read the data meta info file
    meta_fp = read_np_mmap(data_dir, 'meta', 'dat', np.int32, 'r')
    print(f'local_rank: {local_rank}, meta_fp shape: {meta_fp.shape}')
    # print(f'meta_fp: {meta_fp}')
    
    # slice this local rank episode length slice
    eplen_slice = chunk_into_slices(meta_fp.shape[0], world_size)[local_rank] # slice of this local_rank
    print(f'eplen_slice: {eplen_slice}')
    
    # load meta[contain the length of each episode]
    meta = np.zeros((eplen_slice.stop - eplen_slice.start,), dtype=np.int32)
    meta[:] = meta_fp[eplen_slice]
    
    # limit this rank steps <= max_train_steps
    if max_train_steps is not None:
        meta_eplen_counter = 0
        for i, meta_eplen in enumerate(meta):
            meta_eplen_counter += meta_eplen
            if meta_eplen_counter >= max_train_steps:
                meta = meta[:i+1]
                break
    
    # get episode trajectories that are concat together
    pre_ranks_steps = np.sum(meta_fp[:eplen_slice.start])
    this_rank_step = np.sum(meta)
    step_slice = slice(pre_ranks_steps, pre_ranks_steps + this_rank_step)
    print(f"step_slice: {step_slice}")
    src_fp, tgt_fp = list(map(lambda x: read_np_mmap(data_dir, x, 'dat', np.float32, 'r'),
                                ['src', 'tgt']))
    src_dim, tgt_dim = src_fp.shape[-1], tgt_fp.shape[-1]
    src, tgt = list(map(lambda x: np.zeros((step_slice.stop - step_slice.start, x), 
                                            dtype=np.float32), [src_dim, tgt_dim]))
    print(f"reading data from disk...")
    src[:], tgt[:] = src_fp[step_slice], tgt_fp[step_slice]
    
    # split the whole data into episodes
    print(f"spliting data...")
    src_episodes, tgt_episodes = list(map(
        lambda x: split_np2tensorList(x, meta, max_episode_length, 
                                      return_type=return_type),
        [src, tgt],
    ))
    
    # add print info to buffer
    print_info_buf['src_shape'] = src.shape
    print_info_buf['tgt_shape'] = tgt.shape
    print_info_buf['meta_shape'] = meta.shape
    print_info_buf['src_episodes_len'] = len(src_episodes)
    print_info_buf['tgt_episodes_len'] = len(tgt_episodes)
    
    return src_episodes, tgt_episodes

def loadData_from_disk_episode2chunkMode(local_vars: dict, return_type='torch'):
    '''
    load data from disk if the data is in episode mode,
    and convert the episode data to chunk mode
    
    Return:
        srcs: np.ndarray (sample_num, seq_len, state_dim, )
        tgts: np.ndarray (sample_num, seq_len, action_dim, )
    '''
    print("loading data from disk in episode2chunk mode.")
    
    assert return_type in ['numpy', 'torch'], return_type
    
    # load the data from disk
    # src_episodes: list[np.ndarray((eplen, state_dim))]
    # tgt_episodes: list[np.ndarray((eplen, action_dim))]
    src_episodes, tgt_episodes = loadData_from_disk_episodeMode(local_vars, return_type='numpy')
    eplen_episodes = [len(src_episode) for src_episode in src_episodes]
    print("original steps number of eplen_episodes:", sum(eplen_episodes))
    
    # load vars from locals()
    seq_len = local_vars['seq_len']
    dt_mode = local_vars['dt_mode']
    max_sample_number = local_vars['max_sample_number']
    state_dim = src_episodes[0].shape[-1]
    action_dim = tgt_episodes[0].shape[-1]
    src_dim, tgt_dim = parse_dt_mode_to_src_tgt_dim(dt_mode, state_dim, action_dim)
    src_mode, tgt_mode = dt_mode.split('_')
    
    # srcs: np.ndarray (sample_num, seq_len, state_dim, )
    # tgts: np.ndarray (sample_num, seq_len, action_dim, )
    srcs, tgts = convert_episodes_to_samples_nonmdp_numba(
        src_episodes, tgt_episodes, eplen_episodes, seq_len, 0, max_sample_number,
        state_dim, action_dim, src_dim, tgt_dim, src_mode, tgt_mode)
    print(f"srcs.shape: {srcs.shape}, tgts.shape: {tgts.shape}")
    
    # convert srcs and tgts to the right return type
    if return_type == 'torch':
        srcs = torch.from_numpy(srcs)
        tgts = torch.from_numpy(tgts)
    
    return srcs, tgts, src_episodes, tgt_episodes

def get_dataloaders(
    data_read_from, load_data_mode: str, tasks: list, seq_len: int, dt_mode: str,
    aligo_name: str, aligo_data_root_dir: str, local_rank, world_size, train_ratio,
    dtype, aligo_enable, batch_size: int, data_scale: bool, distributed: bool,
    data_mode: str, max_sample_number: int, load_data_statistics: bool,
    task_config: dict, device
):
    '''
        @data_read_from: e.g. ../../../../share/FCNet/data
        @src: [sample_num, seq_len, patch_num * tick_dim]
        @tgt: [sample_num,](categories according to ATR)
        @task_config: dict(
            <task_name>: dict(
                player_level: 'expert'
                max_episode_length: 256
                max_train_steps: 60_000_000
            )
        )
    '''
    data_files_in_tasks = check_data_validity(
        aligo_name, aligo_data_root_dir, data_read_from, 
        tasks, aligo_enable, data_mode, task_config
    )
    src_ds, tgt_ds = [], [] # state or action dim of src and tgt
    srcs, tgts, train_srcs, test_srcs, train_tgts, test_tgts, src_episodes_list, tgt_episodes_list = \
        [], [], [], [], [], [], [], []
    dataset_info_buf, print_info_buf = dict(), dict()
    for task, data_file in zip(tasks, data_files_in_tasks):
        print_info_buf['data_file'] = data_file
        print_info_buf['local_rank'] = local_rank
        
        # extract info from task_config
        max_episode_length = task_config[task]['max_episode_length']
        max_train_steps = task_config[task]['max_train_steps']
        
        # data in memory is chunk
        if load_data_mode in CHUNK_LOAD_MODES:
            # load data from disk
            src, tgt, src_episodes, tgt_episodes = loadData_from_disk_episode2chunkMode(locals())
            srcs.append(src); tgts.append(tgt)
            src_episodes_list.append(src_episodes); tgt_episodes_list.append(tgt_episodes)
            
            # split episode dataset
            train_src, test_src = split_train_test_1d(src, train_ratio)
            train_tgt, test_tgt = split_train_test_1d(tgt, train_ratio)
            
            # add this task's [train, test] * [src, tgt] to the list of multi tasks
            for k in ['train_src', 'test_src', 'train_tgt', 'test_tgt']:
                eval(f'{k}s').append(eval(k))
            
            # add src and tgt's dimension
            src_ds.append(train_src.shape[-1])
            tgt_ds.append(train_tgt.shape[-1])
            
            # print run info
            print_info_buf_exec(print_info_buf)
            
            
            # # load data from disk
            # if load_data_mode == 'chunk':
            #     src, tgt = loadData_from_disk_chunkMode(locals())
            # elif load_data_mode == 'episode2chunk':
            #     src, tgt = loadData_from_disk_episode2chunkMode(locals())
            # print_info_buf['src_shape'] = src.shape
            # print_info_buf['tgt_shape'] = tgt.shape
            
            # # count nan, max, min
            # if load_data_statistics:
            #     print_info_buf['nan_cnt'] = src.isnan().sum().item() + tgt.isnan().sum().item()
            #     print_info_buf['src_max'] = src.max().item()
            #     print_info_buf['tgt_max'] = tgt.max().item()
            #     print_info_buf['src_min'] = src.min().item()
            #     print_info_buf['tgt_min'] = tgt.min().item()
            
            # # split the whole data into train data and test data
            # train_src, test_src = split_train_test_1d(src, train_ratio)
            # train_tgt, test_tgt = split_train_test_1d(tgt, train_ratio)
            # train_srcs.append(train_src); test_srcs.append(test_src)
            # train_tgts.append(train_tgt); test_tgts.append(test_tgt)
            # src_seq_len, src_dim, tgt_seq_len, tgt_dim = max(src_seq_len, train_src.shape[-2]), \
            #     max(src_dim, train_src.shape[-1]), max(tgt_seq_len, train_tgt.shape[-2]), \
            #     max(tgt_dim, train_tgt.shape[-1])

            # # print run info
            # print_info_buf_exec(print_info_buf)
        
        # data in memory is episode
        elif load_data_mode in EPISODE_LOAD_MODES:
            raise NotImplementedError

        else:
            raise NotImplementedError
    
    # memory data is chunk mode
    if load_data_mode in CHUNK_LOAD_MODES:
        # define data collator
        src_d, tgt_d = max(src_ds), max(tgt_ds)
        my_collator = ChunkCollator(src_d, tgt_d) # custom stack different len tasks data
        
        # create datasets and dataset's length of different gpus
        train_dataset, test_dataset = ChunkDataset(train_srcs, train_tgts), \
            ChunkDataset(test_srcs, test_tgts)
        synced_train_len, synced_test_len = [partial(sync_mgpu, device=device, opt='min')(x=len(k)) \
            for k in [train_dataset, test_dataset]]
        train_dataset.total_length, test_dataset.total_length = synced_train_len, synced_test_len
        
        # create dataloaders
        train_loader, test_loader = \
            make_dataloader(train_dataset, my_collator=my_collator, batch_size=batch_size), \
            make_dataloader(test_dataset, my_collator=my_collator, batch_size=batch_size)
    
    # memory data is episode mode
    elif load_data_mode in EPISODE_LOAD_MODES:
        raise NotImplementedError('episode mode not support yet')
        # print('creating datasets...')
        # train_dataset, test_dataset = EpisodeDataset(src_episodes_train, tgt_episodes_train), \
        #     EpisodeDataset(src_episodes_test, tgt_episodes_test)
        # print('creating dataloaders...')
        # train_loader, test_loader = \
        #     EpisodeLoader(train_dataset, batch_size, shuffle = True, device = device, distributed = distributed), \
        #     EpisodeLoader(test_dataset, batch_size, shuffle = True, device = device, distributed = distributed)
        
        # count real dataloader length
        # print('counting real dataloader length...')
        # real_trainloader_len, real_testloader_len = 0, 0
        # for foo_data in train_loader:
        #     real_trainloader_len += 1
        # for foo_data in test_loader:
        #     real_testloader_len += 1
        # print(f"real_trainloader_len: {real_trainloader_len}, real_testloader_len: {real_testloader_len}")
    else:
        raise NotImplementedError
    
    print(f"train dataset len: {len(train_dataset)}, test dataset len: {len(test_dataset)}")
    print(f"train dataloader len: {len(train_loader)}, test dataloader len: {len(test_loader)}")
    return train_loader, test_loader, src_d, tgt_d, dataset_info_buf, srcs, tgts, src_episodes_list, tgt_episodes_list

    # return (
    #     [Dataset_mk(train_srcs[i], train_tgts[i]) for i in range(len(train_srcs))],
    #     [Dataset_mk(test_srcs[i], test_tgts[i]) for i in range(len(test_srcs))],
    #     src_dim,
    #     tgt_dim,
    # )
