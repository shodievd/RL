import torch
import torch.nn as nn
import copy

from .utils import mlp, DEFAULT_DEVICE, polyak_avg, asymmetric_l2_loss


EXP_ADV_MAX = 100.

# Critic class
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

# Actor class
class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=[256], n_hidden=2):
        super().__init__()
        self.net = mlp([obs_dim] + hidden_dim * n_hidden + [act_dim], 
                       output_activation=nn.Tanh)

    def forward(self, obs):
        return self.net(obs)
    
    def act(self, obs, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            return self(obs)

class IQL(nn.Module):
    def __init__(self, qf, vf, policy, optimizer_factory, 
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
    
    # Implicit Q-Learning
    def update(self, observations, actions, next_observations, 
               rewards, terminals):
        # Calculate some values
        with torch.no_grad():
            target_q = self.q_target(observations, actions) # For L_v
            next_v = self.vf(next_observations)             # For L_q
        
        # Update Value function
        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau)
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        # Update Q function
        targets = rewards + (1 - terminals.float()) * self.gamma * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(nn.functional.mse_loss(q, targets) for q in qs)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        polyak_avg(self.q_target, self.qf, self.polyak)

        # Update policy
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.policy(observations)
        bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        policy_loss = torch.mean(exp_adv * bc_losses)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
