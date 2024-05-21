import os
from os.path import join, exists
import shutil
import pickle
import random
import functools

import numpy as np

from FCNet.DT.utils.common import clear_components, save_np_mmap
from FCNet.DT.utils.data_utils import set_epData_mode
from FCNet.DT.utils.name_utils import EPISODE_SAVE_DATA_FORMAT

from d4rl_package_funcs import normalize

# locate at FCNet/DT/d4rl
src_data_dir = './d4rl-data' # contains pkls
tgt_data_dir = '../data/data' # contains dats which is 

# env_names = ['ant', 'halfcheetah', 'hopper', 'walker2d']
# datasets = ['medium', 'medium-replay', 'medium-expert', 'expert']
# env_names = ["pen", "hammer", "relocate"]  # "door", 
# datasets = ["human", "cloned", "expert"]
env_names = ["kitchen"]  # "door", 
datasets = ["complete", "partial", "mixed"]

def merge_rew_with_state(state: np.ndarray, rew: np.ndarray, norm_func):
    '''
    Args:
        state: (steps, state_dim)
        rew: (steps, )
    
    Returns:
        state: (steps, state_dim+1)
    '''
    assert state.shape[0] == rew.shape[0]
    assert state.shape[0] > 0
    # calculate suffix of reward
    rew = rew.copy()
    for i in range(rew.shape[0]-2, -1, -1):
        rew[i] += rew[i+1]
    
    rew = norm_func(rew)
    
    # merge
    state = np.concatenate([np.expand_dims(rew, axis=1), state], axis=1)
    
    return state

def main():
    for env_name in env_names:
        for dataset in datasets:
            # total_collect_samples means episode cnt
            task, player_level, total_collect_samples, max_episode_length = env_name, dataset, \
                0, -np.inf
            
            pkl_path = join(src_data_dir, f'{env_name}-{dataset}-v0.pkl')
            state_episodes, action_episodes = list(), list()
            with open(pkl_path, 'rb') as f:
                trajectories = pickle.load(f)
            random.shuffle(trajectories)
            
            the_task_steps = 0 # total steps of this task, for assert
            for trajectory in trajectories:
                state_episodes.append(
                    merge_rew_with_state(trajectory['observations'], 
                                         trajectory['rewards'], 
                                         functools.partial(normalize, f"{env_name}-{dataset}"))
                )
                # state_episodes.append(trajectory['observations'])
                action_episodes.append(trajectory['actions'])
                the_task_steps += state_episodes[-1].shape[0]
                total_collect_samples += 1
                max_episode_length = max(max_episode_length, trajectory['observations'].shape[0])
            state_episodes, action_episodes, episodes_length = \
                set_epData_mode((state_episodes, action_episodes), 'assemble', 'numpy')
            state_episodes, action_episodes, episodes_length = \
                state_episodes.astype(np.float32), action_episodes.astype(np.float32), episodes_length.astype(np.int32)
            
            # inconsistency dtype will result in error
            # e.g.
            # episodes_length in memory: [1000 1000 1000 1000 ...] np.int64
            # episodes_length in disk: [1000 0 1000 0 ...] np.int64
            
            assert the_task_steps == state_episodes.shape[0]
            assert the_task_steps == action_episodes.shape[0]
            assert the_task_steps == np.sum(episodes_length)
            
            data_name = eval(EPISODE_SAVE_DATA_FORMAT)
            clear_components(join(tgt_data_dir, data_name))
            
            file_format = 'dat'
            save_np_mmap(state_episodes, join(tgt_data_dir, data_name), 'src', file_format)
            save_np_mmap(action_episodes, join(tgt_data_dir, data_name), 'tgt', file_format)
            save_np_mmap(episodes_length, join(tgt_data_dir, data_name), 'meta', file_format)

if __name__ == '__main__':
    main()