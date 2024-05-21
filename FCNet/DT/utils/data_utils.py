from typing import Union
import random
import numpy as np
import torch


def shuffle_episodes(state_episodes: np.ndarray, action_episodes: np.ndarray, episodes_length: np.ndarray):
    '''
    shuffle episodes format data
    @state_episodes: (ep_cnt * avg_ep_len, state_dim, )
    @action_episodes: (ep_cnt * avg_ep_len, action_dim, )
    @episodes_length: (ep_cnt, )
    '''
    x, y, z = np.zeros_like(state_episodes), np.zeros_like(action_episodes), np.zeros_like(episodes_length)
    acc_eplen, slices = 0, []
    for eplen in episodes_length:
        slices.append((eplen, slice(acc_eplen, acc_eplen + eplen)))
        acc_eplen += eplen
    random.shuffle(slices)
    acc_eplen = 0
    for i, (eplen, epslice) in enumerate(slices):
        x[acc_eplen: acc_eplen + eplen, :] = state_episodes[epslice, :]
        y[acc_eplen: acc_eplen + eplen, :] = action_episodes[epslice, :]
        z[i] = eplen
        acc_eplen += eplen
    return x, y, z

def convert_data_to_tgtReturnType(data: Union[tuple, list, dict, np.ndarray, torch.Tensor], return_type:str):
    '''
    convert the whole data to torch.Tensor or numpy.ndarray.
    @return_type: torch, numpy
    '''
    if isinstance(data, (tuple, list)):
        res = list()
        for x in data:
            res.append(convert_data_to_tgtReturnType(x, return_type))
        if isinstance(data, tuple):
            res = tuple(res)
    elif isinstance(data, dict):
        res = dict()
        for k, v in data.items():
            res[k] = convert_data_to_tgtReturnType(v, return_type)
    elif isinstance(data, np.ndarray) and return_type == 'torch':
        res = torch.from_numpy(data)
    elif isinstance(data, torch.Tensor) and return_type == 'numpy':
        res = data.numpy()
    else:
        res = data
    
    return res

# episode data has two modes: assemble mode, discrete mode
# used for set data type on disk
def set_epData_mode(data_tuple: tuple, tgt_mode: str, return_type:str='numpy'):
    '''
    @data_tuple: (state_episodes: np.ndarray(ep_cnt * avg_ep_len, state_dim, ), 
                  action_episodes: np.ndarray(ep_cnt * avg_ep_len, action_dim, ), 
                  episodes_length: (ep_cnt, )) # assemble mode
                  OR
                 (state_episodes: list[np.ndarray(ep_cnt, state_dim, )],
                  action_episodes: list[np.ndarray(ep_cnt, action_dim, )]) # discrete mode
    @tgt_mode: assemble, discrete
    @return_type: numpy, torch
    @return: solved data tuple
    '''
    assert tgt_mode in ['assemble', 'discrete'], tgt_mode
    assert isinstance(data_tuple, tuple), f'{data_tuple}, {type(data_tuple)}'
    assert len(data_tuple) in [2, 3], len(data_tuple)
    
    now_mode = 'assemble' if len(data_tuple) == 3 else 'discrete'
    
    if now_mode == tgt_mode:
        return data_tuple
    
    if now_mode == 'assemble':
        state_episodes, action_episodes, episodes_length = data_tuple
        now_pos = 0
        state_episodes_res, action_episodes_res = list(), list() # discrete mode data list
        for eplen in episodes_length:
            state_episodes_res.append(state_episodes[now_pos: now_pos+eplen, ...])
            action_episodes_res.append(action_episodes[now_pos: now_pos+eplen, ...])
        tgt_data_tuple = convert_data_to_tgtReturnType((state_episodes_res, action_episodes_res), return_type)
    elif now_mode == 'discrete':
        state_episodes, action_episodes = data_tuple
        
        # count total steps
        total_steps = 0
        for state_episode in state_episodes:
            total_steps += state_episode.shape[0]
        
        # create tgt data memory space
        state_dim = state_episodes[0].shape[-1]
        action_dim = action_episodes[0].shape[-1]
        backend = np if isinstance(state_episodes[0], np.ndarray) else torch
        state_episodes_res, action_episodes_res, episodes_length_res = \
            backend.zeros((total_steps, state_dim), dtype=state_episodes[0].dtype), \
            backend.zeros((total_steps, action_dim), dtype=action_episodes[0].dtype), \
            backend.zeros((len(state_episodes)), dtype=backend.int64)
        
        # fill data
        now_pos = 0
        for i, (state_episode, action_episode) in enumerate(zip(state_episodes, action_episodes)):
            state_episodes_res[now_pos: now_pos+state_episode.shape[0], ...] = state_episode[:]
            action_episodes_res[now_pos: now_pos+action_episode.shape[0], ...] = action_episode[:]
            episodes_length_res[i] = action_episode.shape[0]
            now_pos += action_episode.shape[0]
        
        tgt_data_tuple = convert_data_to_tgtReturnType((state_episodes_res, action_episodes_res, episodes_length_res), return_type)
    
    return tgt_data_tuple
    