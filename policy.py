import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math


def one_hot_encode(actions, num_actions, device):
    one_hot_actions = torch.zeros(len(actions), num_actions, dtype=torch.float, device=device)
    one_hot_actions[torch.arange(len(actions)), actions] = 1
    return one_hot_actions


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class DQN(nn.Module):

    def __init__(self, input_dim, output_dim, hidden1_dim=256, hidden2_dim=512):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.fc3 = nn.Linear(hidden2_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class Actor(DQN):

    def __init__(self, input_dim, output_dim, hidden1_dim=256, hidden2_dim=512):
        super(Actor, self).__init__(input_dim, output_dim, hidden1_dim, hidden2_dim)

    def forward(self, x, state_trajs=None, action_trajs=None, ts_trajs=None, train=False):
        return torch.tanh(super().forward(x))


class Critic(DQN):

    def __init__(self, input_dim, output_dim, action_dim, hidden1_dim=256, hidden2_dim=512):
        super(Critic, self).__init__(input_dim, output_dim, hidden1_dim, hidden2_dim)
        self.fc2 = nn.Linear(hidden1_dim + action_dim, hidden2_dim)

    def forward(self, x, a=None, state_trajs=None, action_trajs=None, ts_trajs=None, train=False):
        assert a is not None
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(torch.cat((x, a), dim=-1)))
        return self.fc3(x)


class PolicyBase:
    def __init__(self, state_dim, action_dim, device, gamma=0.99, latent=False):
        self.device = device
        self.num_states = state_dim
        self.num_actions = action_dim
        self.gamma = gamma
        self.latent = latent

    def __repr__(self):
        return "Base"

    def select_action(self, state, eps=None):
        raise NotImplementedError

    def optimize(self, memory, Transition, rms=None):
        pass

    def update_target_policy(self):
        pass
