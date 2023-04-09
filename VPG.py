import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from torch.distributions.categorical import Categorical
from torch.nn import MSELoss


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []

    for i in range(len(sizes) - 1):
        linear = nn.Linear(sizes[i], sizes[i + 1])
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [linear, act()]
    return nn.Sequential(*layers)


def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i + 1] if i < n - 1 else 0)
    return rtgs


def advantage_estimate(values, rtgs, gamma):
    T = len(values)
    advtgs = np.zeros_like(values)
    for t in range(T):
        rew_sum = 0
        for k in range(0, T - t - 1):
            rew_sum += (gamma ** k) * rtgs[k + t]
        advtgs[t] = -values[t] + rew_sum
    return advtgs


def train(env_name='CartPole-v1', hidden_sizes=[32], gamma=0.99, lr=1e-3,
          epochs=50, batch_size=5000, render=False):
    env = gym.make(env_name, render_mode='human')
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    policy_net = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])
    value_func_net = mlp(sizes=[obs_dim] + hidden_sizes + [1])

    policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    value_optimizer = torch.optim.Adam(value_func_net.parameters(), lr=lr)
    print(epochs)

    def get_policy(obs):
        logits = policy_net(obs)
        return Categorical(logits=logits)

    def get_action(obs):
        return get_policy(obs).sample().item()

    def policy_loss_compute(obs, acts, advtgs):
        logp = get_policy(obs).log_prob(acts)
        return -(logp * advtgs).mean()

    def value_loss_compute(obs, weights):
        return ((value_func_net(obs) - weights) ** 2).mean()

    def train_one_epoch():
        batch_obs = []
        batch_acts = []
        batch_weights = []  # rewards-to-go
        batch_rets = []  # epoch return
        batch_lens = []
        batch_advs = []

        obs = env.reset()[0]
        done = False  # signal that episode is completed
        ep_rews = []  # for rewards accured throughout episode
        ep_values = []
        ep_obs = []
        finished_rendering_this_epoch = False

        while True:

            if (not finished_rendering_this_epoch) and render:
                env.render()

            batch_obs.append(obs)
            ep_obs.append(obs)
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            value = value_func_net(torch.as_tensor(obs, dtype=torch.float32))
            ep_values.append(value.item())
            obs, rew, done, _, __ = env.step(act)

            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                rtg = reward_to_go(ep_rews)
                batch_weights += list(rtg)
                batch_advs += list(advantage_estimate(ep_values, batch_weights, gamma=gamma))

                value_optimizer.zero_grad()
                value_loss = value_loss_compute(obs=torch.as_tensor(ep_obs, dtype=torch.float32),
                                                weights=torch.as_tensor(list(rtg), dtype=torch.float32)
                                                )
                value_loss.backward()
                value_optimizer.step()

                obs, done, ep_rews, ep_values, ep_obs = env.reset()[0], False, [], [], []

                finished_rendering_this_epoch = True

                if len(batch_obs) > batch_size:
                    break

        policy_optimizer.zero_grad()
        policy_batch_loss = policy_loss_compute(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                                acts=torch.as_tensor(batch_acts, dtype=torch.float32),
                                                advtgs=torch.as_tensor(batch_advs, dtype=torch.float32)
                                                )
        policy_batch_loss.backward()
        policy_optimizer.step()

        return policy_batch_loss, batch_rets, batch_lens

    for i in range(epochs):
        policy_batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss_p: %.3f \t return: %.3f \t ep_len: %.3f' %
              (i, policy_batch_loss, np.mean(batch_rets), np.mean(batch_lens)))


train(render=False, epochs=100)
