import gym
import d4rl
import numpy as np
import torch
from tqdm import tqdm

from .IQL import IQL, DeterministicPolicy, TwinQ, ValueFunction
from .utils import set_seed, sample_batch, torchify, get_env_dataset, evaluate_policy


def train(env_name, tau, beta, seed=1, n_steps=1e6, lr=1e-3, 
          batch_size=256, eval_period=5000, n_eval_episodes=10, max_episode_steps=1000):
    
    env, dataset = (env_name)
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]
    set_seed(seed, env)

    policy = DeterministicPolicy(obs_dim, act_dim)

    def eval_policy():
        eval_returns = np.array([evaluate_policy(env, policy, max_episode_steps) for _ in range(n_eval_episodes)])
        normalized_returns = d4rl.get_normalized_score(env_name, eval_returns) * 100.0

        print(f"return mean: {eval_returns.mean():.3}, return std: {eval_returns.std():.3}")
        print(f"normilized return mean: {normalized_returns.mean():.3}, normilized return std: {normalized_returns.std():.3}")

    iql = IQL(
        qf=TwinQ(obs_dim, act_dim),
        vf=ValueFunction(obs_dim, act_dim),
        policy=policy
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=lr),
        tau=tau,
        beta=beta
    )

    for step in tqdm(range(n_steps)):
        iql.update(**sample_batch(dataset, batch_size))

        if (step + 1) % eval_period == 0:
            eval_policy()

    torch.save(iql.state_dict(), "./")  
    