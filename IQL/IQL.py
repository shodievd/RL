import torch
import torch.nn as nn
from .utils import mlp, DEFAULT_DEVICE, polyak_avg
import copy

class TwinQ(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = mlp(dims, squeeze_output=True)
        self.q2 = mlp(dims, squeeze_output=True)

    def both(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state, action):
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = mlp(dims, squeeze_output=True)

    def forward(self, state):
        return self.v(state)

class DeterministicPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=[256], n_hidden=2):
        super().__init__()
        self.net = mlp(list(obs_dim) + hidden_dim * n_hidden + list(act_dim), 
                       output_activation=nn.Tanh)

    def forward(self, obs):
        return self.net(obs)
    
    def act(self, obs, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            return self(obs)

class IQL(nn.Module):
    def __init__(self, qf, vf, policy, optimizer_factory, max_steps, 
                 tau, beta, gamma=0.99, polyak=0.995):
        super().__init__()
        self.qf = qf.to(DEFAULT_DEVICE)
        self.q_target = copy.deepcopy(qf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.vf = vf.to(DEFAULT_DEVICE)
        self.policy = policy.to(DEFAULT_DEVICE)
        self.v_optimizer = optimizer_factory(self.vf.parameters())
        self.q_optimizer = optimizer_factory(self.qf.parameters())
        self.policy_optimizer = optimizer_factory(self.policy.parameters())
        self.tau = tau
        self.beta = beta
        self.gamma = gamma
        self.polyak = polyak
    
    def update(self, observations, actions, next_observations, 
               rewards, terminals):
        pass
    #Катка Дотки под баночку пива и сразу доделать =)