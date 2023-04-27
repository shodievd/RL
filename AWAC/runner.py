import gym
import d4rl
import numpy as np
import torch
from tqdm import tqdm
import os
import copy

from .awac import AWAC, Policy, TwinQ
from .utils import set_seed, sample_batch, torchify, get_env_dataset, evaluate_policy


def AWACRunner(experiment_name: str, env_name: str = 'halfcheetah-medium-v2', awac_lambda: float = 1., seed: int = 1, n_steps: int = 10**6, lr: float = 1e-3, 
          batch_size: int = 256, eval_period:int = 5000, n_eval_episodes: int = 10, max_episode_steps: int = 1000):
    
    env, dataset = get_env_dataset(env_name, max_episode_steps)
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]
    set_seed(seed, env)
    best_rew = 0.
    policy = Policy(obs_dim, act_dim)

    def eval_policy() -> float:
        eval_returns = np.array([evaluate_policy(env, policy, max_episode_steps) for _ in range(n_eval_episodes)])
        normalized_returns = d4rl.get_normalized_score(env_name, eval_returns) * 100.0

        print('\n')
        print("=======================")
        print(f"return | mean: {eval_returns.mean():.3}, std: {eval_returns.std():.3}")
        print(f"normilized | mean: {normalized_returns.mean():.3}, std: {normalized_returns.std():.3}")
        print("=======================")
        return normalized_returns.mean()
    
    awac_alg = AWAC(
        qf=TwinQ(obs_dim, act_dim),
        policy=policy,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=lr),
        awac_lambda=awac_lambda
    )
    best_wts = copy.deepcopy(awac_alg.state_dict())

    for step in tqdm(range(n_steps)):
        awac_alg.update(**sample_batch(dataset, batch_size))

        if (step + 1) % eval_period == 0:
            ret = eval_policy()
            if ret > best_rew:
                best_wts = copy.deepcopy(awac_alg.state_dict())
    
    os.makedirs(experiment_name)
    torch.save(best_wts, os.path.abspath('.') + '/' + experiment_name + '/awac.pth')  
    print("Training completed!")
