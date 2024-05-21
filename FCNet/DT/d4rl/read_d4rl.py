import gym
import gymnasium
import numpy as np

import collections
import pickle
from d4rl.ope import normalize

import matplotlib.pyplot as plt
import d4rl

data_path = "./d4rl-data"
def plot_time_and_energy(x, info=""):
    """
    x: (N, dim)
    Plot the energy of the signal x(t) in the frequency domain.
    """
    plt.clf()
    # time_domain_y_label = "Mass Position"
    # time_domain_y_label = 'Hip Motor Position(sin)'
    time_domain_y_label = 'Motor Velocity'

    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'black', 'yellow']
    seq_len = x.shape[0]
    n_modes = 10 # int(2.5 * np.log(seq_len))
    dim = x.shape[1]
    # plot original signal
    for i in range(dim):
        plt.plot(x[:, i], color=colors[i], label=f'original signal of {i}')
    plt.xlabel('Time')
    # plt.ylabel(time_domain_y_label)
    plt.ylabel(time_domain_y_label)
    plt.legend()
    plt.savefig(f'plot_original_signal_{info}.pdf')
    plt.clf()
    
    # y is truncated to 0:n_modes in the frequency domain from the original signal x
    y = np.zeros((seq_len, dim))
    for i in range(dim):
        # Compute the FFT
        fft = np.fft.rfft(x[:, i]) # rfft
        
        y[:, i] = np.fft.irfft(fft[:n_modes], n=seq_len)  # irfft
        
        # Compute the energy
        energy = np.abs(fft) ** 2  # (128)
        # Plot the energy
        # moving average
        energy = np.convolve(energy, np.ones(20), 'valid') / 20  # (109)
        print(f"energy.shape: {energy.shape}")
        plt.plot(energy, color=colors[i], label=f'energy of {i}')
    # Draw a vertical dotted line in n_modes
    plt.axvline(x=n_modes, linestyle='--', color='gray')
    plt.xlabel('Frequency Mode')
    plt.ylabel('Energy Density')
    plt.legend()
    # plt.yscale('log')
    plt.savefig(f'plot_energy_{info}.pdf')
    # plt.savefig(f'plot/plot_energy_{info}.png')
    # plt.show()
    plt.clf()
    
    # plot y
    for i in range(dim):
        plt.plot(y[:, i], color=colors[i], label=f'transformed signal of {i}')
    plt.xlabel('Time')
    plt.ylabel(time_domain_y_label)
    plt.legend()
    plt.savefig(f'plot_transformed_signal_{info}.pdf')
    
for dataset_type in ["complete", "partial", "mixed"]:  # ["medium", "medium-replay", "expert", "medium-expert"]
    print(f"# ------------------------------------{dataset_type}--------------------------------")
    for env_name in ["kitchen"]: # ant ["halfcheetah", "hopper", "walker2d"]
        print(f"# ------------------------------------{env_name}--------------------------------")
        name = f"{env_name}-{dataset_type}-v0"
            
        with open(f"{data_path}/{name}.pkl", "rb") as f:
            paths = pickle.load(f)
        
        # state and action dim
        # print(f"State dim: {paths[0]['observations'][0].shape}")  # paths[0]["observations"]: (lengths, state_dim)
        # print(f"Action dim: {paths[0]['actions'][0].shape}")
        # length of state, action, next_state, reward
        # print(f"State length: {paths[0]['observations'].shape[0]}")
        # print(f"Action length: {paths[0]['actions'].shape[0]}")
        # print(f"Next state length: {paths[0]['next_observations'].shape[0]}")
        # print(f"Reward length: {paths[0]['rewards'].shape[0]}")
        # print("last state", paths[0]['observations'][-2], paths[0]['observations'][-1])
        # print("last next state", paths[0]['next_observations'][-2], paths[0]['next_observations'][-1])
        
        print(f"Number of trajectories: {len(paths)}")
        # length of each trajectory
        lengths = [p["rewards"].shape[0] for p in paths]
        print(
            f"Trajectory lengths: mean = {np.mean(lengths)}, std = {np.std(lengths)}, max = {np.max(lengths)}, min = {np.min(lengths)}"
        )
        print(f"LONG length trajectories number: {np.sum(np.array(lengths) > 128)}")
        total_sample_num = 0
        seq_len = 64
        for p in paths:
            eplen = p["rewards"].shape[0]
            if eplen >= seq_len:
                total_sample_num += 1
            if eplen > seq_len:
                total_sample_num += (eplen - ((eplen - 1) % seq_len + 1)) // seq_len
        print("total_sample_num * seq_len", total_sample_num, seq_len)
        
        # plot_time_and_energy(paths[0]["observations"][:1000], "hopper_medium")
        
        returns = np.array([np.sum(p["rewards"]) for p in paths])
        selected_normalized_returns = []
        for r in returns[::10]:
           selected_normalized_returns.append(normalize(env_name + "-" + dataset_type, r))
        # print("selected_normalized_returns", selected_normalized_returns)
              
        num_samples = np.sum([p["rewards"].shape[0] for p in paths])
        print(f"Number of samples collected: {num_samples}")
        print(
            f"Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}"
        )

        with open(f"{name}.pkl", "wb") as f:
            pickle.dump(paths, f)