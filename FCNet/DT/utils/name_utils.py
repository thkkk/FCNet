EPISODE_SAVE_DATA_FORMAT = '''\
f'{task}_{player_level}_{int(total_collect_samples)}_{int(max_episode_length)}'\
''' # use eval()

EPISODE_LOAD_DATA_RE = '''\
rf'{task}_{player_level}_(.*)'\
''' # use eval()

# CHUNK_SAVE_DATA_FORMAT = '''\
# f'{task}_expert_{int(total_collect_samples)}_{int(max_episode_length)}'\
# ''' # use eval()

# CHUNK_LOAD_DATA_RE = '''\
# rf'{task}_expert_(.*)'\
# ''' # use eval()
