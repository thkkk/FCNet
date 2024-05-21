import numpy as np
import scipy.stats

def aggregate_iqm(scores: np.ndarray):
    """Computes the interquartile mean across runs and tasks. Ref: https://github.com/google-research/rliable/blob/master/rliable/metrics.py

    Args:
    scores: A matrix of size (`num_runs` x `num_tasks`) where scores[n][m]
        represent the score on run `n` of task `m`.
    Returns:
    IQM (25% trimmed mean) of scores.
    """
    return scipy.stats.trim_mean(scores, proportiontocut=0.25, axis=None)


def evaluate(scores: list):
    """以各种方式衡量轨迹return的好坏

    Args:
        scores (list): 多次推理中各条轨迹的return,形如[10.0, 20.0, -10.0, 0.0]
    """
    returns = np.array(scores)
    return_mean = np.mean(returns)
    return_std = np.std(returns)

    mean_success_reward = returns[returns > 0].mean()  # 成功的轨迹的平均return，已经在外面算过了
    success_rate = (returns > 0).sum() / len(returns)  # 认为>0的轨迹是成功的
    iqm_score = aggregate_iqm(returns)  # 25% trimmed mean，去除最大最小的25%后的平均值
    # mean_success_reward: {mean_success_reward:.3f} success_rate: {success_rate:.3f} 

    num_envs = 2048
    summary_info = "\n"
    
    if len(returns) % num_envs == 0:
        episode_return = []
        for i in range(0, len(returns), num_envs):
            episode_return.append(returns[i:i+num_envs].mean())
        summary_info += f"""episode_return: {np.mean(episode_return):.3f} +- {np.std(episode_return):.3f} """
    summary_info += f"""return_mean: {return_mean:.3f} iqm_score: {iqm_score:.3f}\n"""
    summary_info += f"""list len: {len(returns)}\n"""
    return summary_info, return_mean, mean_success_reward, success_rate, iqm_score