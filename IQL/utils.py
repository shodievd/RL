import numpy as np
import torch
import torch.nn as nn
import random
import gym
import d4rl

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)

# For creating MLP
def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity, squeeze_output=False):
    n = len(sizes)
    layers = []
    for i in range(len(sizes) - 1):
        linear = nn.Linear(sizes[i], sizes[i + 1])
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [linear, act()]
    if squeeze_output:
      assert sizes[-1] == 1
      layers.append(Squeeze(-1))
    
    return nn.Sequential(*layers)

def torchify(x:np.array):
    x = torch.from_numpy(x)
    if x.dtype is torch.float64:
        x = x.float()
    x = x.to(device=DEFAULT_DEVICE)
    return x

# Dataset is dict with dict_keys(['observations',
                                # 'actions', 
                                # 'next_observations', 
                                # 'rewards', 
                                # 'terminals'])
def sample_batch(dataset:dict, batch_size:int):
    obs = list(dataset.keys())[0]
    n, device = len(dataset[obs]), dataset[obs].device
    for v in dataset.values():
        assert len(v) == n, 'Dataset values must have same length'
    indices = torch.randint(low=0, high=n, size=(batch_size,), device=device)
    return {k: v[indices] for k, v in dataset.items()}

def set_seed(seed, env=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)

# P_avg: w <- w * p + (1 - p) * w_t
def polyak_avg(target, source, polyak):
    for p, p_targ in zip(source.parameters(), target.parameters()):
        p_targ.data.mul_(polyak)
        p_targ.data.add_((1 - polyak) * p.data)

def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

def get_env_dataset(env_name, max_episode_steps):
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)

    for k, v in dataset.items():
        dataset[k] = torchify(v)
    
    return env, dataset

def evaluate_policy(env, policy, max_episode_steps):
    obs = env.reset()
    episode_reward = 0.

    for _ in range(max_episode_steps):
        with torch.no_grad():
            action = policy.act(torchify(obs)).cpu().numpy()
        next_obs, reward, done, info = env.step(action)
        episode_reward += reward
        if done:
            break
        else:
            obs = next_obs
    return episode_reward   

class Logger:
    def __init__():
      raise NotImplementedError
