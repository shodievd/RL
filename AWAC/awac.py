import torch
import torch.nn as nn
import copy
import numpy as np
from .utils import mlp, DEFAULT_DEVICE, polyak_avg


EXP_ADV_MAX = 100.

# Critic class
class TwinQ(nn.Module):  
    def __init__(self, obs_dim, action_dim, hidden_dim=256, n_hidden=3):
        super().__init__()
        dims = [obs_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = mlp(dims, squeeze_output=True)
        self.q2 = mlp(dims, squeeze_output=True)

    def both(self, state:torch.Tensor, action:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))

# Actor class
class Policy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256, n_hidden: int = 3,
                 min_log_std: float = -5.0, max_log_std: float = 2.0, min_action: float = -1.0, 
                 max_action: float = 2.0,):
        
        super().__init__()
        dims = [obs_dim, *([hidden_dim] * n_hidden), act_dim]
        self.net = mlp(dims)
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.min_action = min_action
        self.max_action = max_action

    def get_policy(self, observations: torch.Tensor) -> torch.distributions.Distribution:
        mean = self.net(observations)
        log_std = self.log_std.clamp(self.min_log_std, self.max_log_std)
        policy = torch.distributions.Normal(mean, log_std.exp())
        return policy
    def log_prob(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        policy = self.get_policy(observations)
        log_prob = policy.log_prob(actions).sum(-1, keepdim=True)
        return log_prob

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        policy = self.get_policy(observations)
        action = policy.rsample()
        action.clamp_(self._min_action, self._max_action)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob
    
    def act(self, observations: torch.Tensor, enabl_grad: bool = False) -> torch.Tensor:
        with torch.set_grad_enabled(enabl_grad):
            policy = self._get_policy(observations)
            action = policy.sample()
            action.clamp_(self._min_action, self._max_action)
            return action
        
# Advantage Weighted Actor Critic (AWAC)
class AWAC(nn.Module):
    def __init__(self, qf, policy, optimizer_factory, 
                 awac_lambda, gamma=0.99, polyak=0.995):
        super().__init__()
        self.qf = qf.to(DEFAULT_DEVICE)
        self.q_target = copy.deepcopy(qf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.policy = policy.to(DEFAULT_DEVICE)
        self.q_optimizer = optimizer_factory(self.qf.parameters())
        self.policy_optimizer = optimizer_factory(self.policy.parameters())
        self.awac_lambda = awac_lambda
        self.gamma = gamma
        self.polyak = polyak
           
    def update(self, observations: torch.Tensor, actions: torch.Tensor, next_observations: torch.Tensor,
               rewards: torch.Tensor, terminals: bool):
        # Calculate some values
        with torch.no_grad():
            next_actions, _ = self.policy(next_observations)
            qt_next = self.q_target(next_observations, next_actions)
        
        # Update Q function (Critic)
        targets = rewards + self.gamma * (1. - terminals.float()) * qt_next
        qs = self.qf.both(observations, actions)
        q_loss = sum(nn.functional.mse_loss(q, targets) for q in qs)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q function
        polyak_avg(self.q_target, self.qf, self.polyak)

        # Update policy
        with torch.no_grad():
            pi_action, _ = self.policy(observations)
            v = self.qf(observations, pi_action)
            q = self.qf(observations, actions)
            adv = q - v
            exp_adv = torch.exp(adv / self.awac_lambda).clamp(max=EXP_ADV_MAX)
        
        act_log_prob = self.policy.log_prob(observations, actions)
        pi_loss = (-act_log_prob * exp_adv).mean()
        self.policy_optimizer.zero_grad()
        pi_loss.backward()
        self.policy_optimizer.step()
