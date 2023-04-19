import numpy as np
import torch
import torch.nn as nn
import random

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# For creating MLP
def mlp(sizes, activation=nn.Relu, output_activation=nn.Identity):
    n = len(sizes)

    for i in range(len(sizes) - 1):
        linear = nn.Linear(sizes[i], sizes[i + 1])
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [linear, act()]
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
    n, device = len(dataset[obs]), dataset[k].device
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

class Logger:
    pass
