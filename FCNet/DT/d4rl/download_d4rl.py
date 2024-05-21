import gym
import gymnasium
import numpy as np

import collections
import pickle

import d4rl


datasets = []

# for env_name in ["ant"]: # ["halfcheetah", "hopper", "walker2d"]:
#     for dataset_type in ["medium", "medium-replay", "expert", "medium-expert"]:

for env_name in ["kitchen"]: # ["pen", "hammer", "door", "relocate"]:
    for dataset_type in ["complete", "partial", "mixed"]:  #["human", "cloned", "expert"]
        name = f"{env_name}-{dataset_type}-v0"
        env = gym.make(name)
        dataset = env.get_dataset()

        N = dataset["rewards"].shape[0]
        data_ = collections.defaultdict(list)

        use_timeouts = False
        if "timeouts" in dataset:
            use_timeouts = True

        episode_step = 0
        paths = []
        for i in range(N):
            done_bool = bool(dataset["terminals"][i])
            if use_timeouts:
                final_timestep = dataset["timeouts"][i]
            else:
                final_timestep = episode_step == 1000 - 1
            for k in [
                "observations",
                "next_observations",
                "actions",
                "rewards",
                "terminals",
            ]:
                if k in dataset:
                    data_[k].append(dataset[k][i])
                # d4rl Adroit does not have next_observations. And FCNet does't need it.
            if done_bool or final_timestep:
                episode_step = 0
                episode_data = {}
                for k in data_:
                    episode_data[k] = np.array(data_[k])
                paths.append(episode_data)
                data_ = collections.defaultdict(list)
            episode_step += 1

        returns = np.array([np.sum(p["rewards"]) for p in paths])
        num_samples = np.sum([p["rewards"].shape[0] for p in paths])
        print(f"Number of samples collected: {num_samples}")
        print(
            f"Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}"
        )

        with open(f"d4rl-data/{name}.pkl", "wb") as f:
            pickle.dump(paths, f)

# door: (6729 samples(steps))?, 1000000, 1000000