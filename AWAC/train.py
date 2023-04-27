import gym
import d4rl
import numpy as np
import torch
from tqdm import tqdm

from .awac import AWAC, Policy, TwinQ
from .utils import set_seed, sample_batch, torchify, get_env_dataset, evaluate_policy


def train(env_name, tau, awac_lambda, seed=1, n_steps=10**6, lr=1e-3, 
          batch_size=256, eval_period=5000, n_eval_episodes=10, max_episode_steps=1000):
    
    env, dataset = get_env_dataset(env_name, max_episode_steps)
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]
    set_seed(seed, env)

    policy = Policy(obs_dim, act_dim)

    def eval_policy():
        eval_returns = np.array([evaluate_policy(env, policy, max_episode_steps) for _ in range(n_eval_episodes)])
        normalized_returns = d4rl.get_normalized_score(env_name, eval_returns) * 100.0

        print('\n')
        print("=======================")
        print(f"return | mean: {eval_returns.mean():.3}, std: {eval_returns.std():.3}")
        print(f"normilized | mean: {normalized_returns.mean():.3}, std: {normalized_returns.std():.3}")
        print("=======================")

    awac_alg = AWAC(
        qf=TwinQ(obs_dim, act_dim),
        policy=policy,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=lr),
        tau=tau,
        awac_lambda=awac_lambda
    )

    for step in tqdm(range(n_steps)):
        awac_alg.update(**sample_batch(dataset, batch_size))

        if (step + 1) % eval_period == 0:
            eval_policy()
    torch.save(awac_alg.state_dict(), "./")  
    
    print("Training completed!")
