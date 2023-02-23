import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence
from torch.nn.modules.rnn import GRU, GRUCell
from torch.nn.utils.rnn import pack_padded_sequence
from torch.distributions.normal import Normal
from torchdiffeq import odeint as odeint
from policy import Actor, Critic
import utils


class ODEFunc(nn.Module):
    def __init__(self, ode_func_net, nonlinear=None):
        super(ODEFunc, self).__init__()
        self.net = ode_func_net
        self.nonlinear = nonlinear

    def forward(self, t, x):
        """
        Perform one step in solving ODE.
        """
        return self.nonlinear(self.net(x)) if self.nonlinear else self.net(x)


class DiffeqSolver(nn.Module):

    def __init__(self, ode_func, method, odeint_rtol, odeint_atol):
        super(DiffeqSolver, self).__init__()
        self.ode_func = ode_func
        self.ode_method = method
        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol

    def forward(self, first_point, time_steps, odeint_rtol=None, odeint_atol=None, method=None):
        """
            Decode the trajectory through ODE Solver
            @:param first_point, shape [N, D]
                    time_steps, shape [T,]
            @:return predicted the trajectory, shape [N, T, D]
        """
        if not odeint_rtol:
            odeint_rtol = self.odeint_rtol
        if not odeint_atol:
            odeint_atol = self.odeint_atol
        if not method:
            method = self.ode_method
        pred = odeint(self.ode_func, first_point, time_steps,
                      rtol=odeint_rtol, atol=odeint_atol, method=method)  # [T, N, D]
        pred = pred.permute(1, 0, 2)  # [N, T, D]
        assert (torch.mean(pred[:, 0, :] - first_point) < 0.001)  # the first prediction is same with first point
        assert pred.size(0) == first_point.size(0)
        assert pred.size(1) == time_steps.size(0)
        assert pred.size(2) == first_point.size(1)
        return pred


class Encoder_z0_RNN(nn.Module):

    def __init__(self, latent_dim, input_dim, device, hidden_to_z0_units=20, bidirectional=False):
        super(Encoder_z0_RNN, self).__init__()
        self.device = device
        self.latent_dim = latent_dim  # latent dim for z0 and encoder rnn
        self.input_dim = input_dim
        self.hidden_to_z0 = nn.Sequential(
            nn.Linear(2 * latent_dim if bidirectional else latent_dim, hidden_to_z0_units),
            nn.Tanh(),
            nn.Linear(hidden_to_z0_units, 2 * latent_dim))
        self.rnn = GRU(input_dim, latent_dim, batch_first=True, bidirectional=bidirectional).to(device)

    def forward(self, data, time_steps, lengths):
        """
            Encode the mean and log variance of initial latent state z0
            @:param data, shape [N, T, D]
                    time_steps, shape [N, T]
                    lengths, shape [N,]
            @:return mean, logvar of z0, shape [N, D_latent]
        """
        data_packed = pack_padded_sequence(data, lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(data_packed)
        assert hidden.size(1) == data.size(0)
        assert hidden.size(2) == self.latent_dim

        # check if bidirectional
        if hidden.size(0) == 1:
            hidden = hidden.squeeze(0)
        elif hidden.size(0) == 2:
            hidden = torch.cat((hidden[0], hidden[1]), dim=-1)
        else:
            raise ValueError('Incorrect RNN hidden state.')

        # extract mean and logvar
        mean_logvar = self.hidden_to_z0(hidden)
        assert mean_logvar.size(-1) == 2 * self.latent_dim
        mean, logvar = mean_logvar[:, :self.latent_dim], mean_logvar[:, self.latent_dim:]
        return mean, logvar


class BaseVAEModel(nn.Module):
    """
        Base VAE model as an abstract class
    """

    def __init__(self, input_dim, latent_dim, eps_decay, encoder_z0, decoder, timer, z0_prior, device):
        super(BaseVAEModel, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device
        self.encoder_z0 = encoder_z0
        self.decoder = decoder
        self.timer = timer
        self.z0_prior = z0_prior
        self.eps = 1.
        self.eps_decay = eps_decay
        self.i_step = 0
        self.criterion = nn.MSELoss()

    def __repr__(self):
        return "BaseVAEModel"

    def encode_next_latent_state(self, data, latent_state, dts):
        """
            predict the next latent state based on the input and the last latent state
            @:param data, shape [N, D]
                    latent_state, shape [N, D_latent]
                    dts, shape [N,]
            @:return shape [N, D_latent]
        """
        raise NotImplementedError("Abstract class cannot be used.")

    def encode_latent_traj(self, states, actions, time_steps, lengths, train=True):
        """
            Encode latent trajectories given states, actions and timesteps
            @:param states, shape [N, T, D_state]
                    actions, shape [N, T, D_action]
                    time_steps, shape [N, T+1]
                    lengths, shape [N,]
            @:return hs, shape [N, T+1, D_latent]
        """
        N = states.size(0)
        if train:
            # encoding
            means_z0, logvars_z0 = self.encoder_z0(torch.cat((states, actions), dim=-1), time_steps, lengths)

            # reparam
            stds_z0 = torch.exp(0.5 * logvars_z0)
            eps = torch.randn_like(stds_z0)
            z0s = means_z0 + eps * stds_z0  # [N, D_latent]
        else:
            means_z0, stds_z0 = None, None
            z0s = self.sample_init_latent_states(num_trajs=N)

        if len(time_steps) == 0:
            return z0s, means_z0, stds_z0

        # solve trajectory
        zs = [z0s]
        for i in range(time_steps.size(1) - 1):
            data = torch.cat((states[:, i, :], actions[:, i, :]), dim=-1)  # [N, D_state+D_action]
            zs.append(self.encode_next_latent_state(data, zs[-1], time_steps[:, i + 1] - time_steps[:, i]))

        return zs, means_z0, stds_z0

    def decode_latent_traj(self, zs):
        """
            Decode latent trajectories
            @:param zs, shape [N, T, D_latent]
            @:return shape [N, T, D_state]
        """
        return self.decoder(zs)
    
    def sample_init_latent_states(self, num_trajs=0):
        shape = (self.latent_dim,) if num_trajs == 0 else (num_trajs, self.latent_dim)
        return self.z0_prior.sample(sample_shape=shape).squeeze(-1)


class LatentODE(BaseVAEModel):
    """
        Latent ODE
    """

    def __init__(self, input_dim, latent_dim, eps_decay, encoder_z0, decoder, diffeq_solver, timer, z0_prior, device):
        super(LatentODE, self).__init__(input_dim, latent_dim, eps_decay, encoder_z0, decoder, timer, z0_prior, device)
        self.diffeq_solver = diffeq_solver
        self.aug_layer = nn.Linear(input_dim + latent_dim, latent_dim).to(device)

    def __repr__(self):
        return "LatentODE"

    def encode_next_latent_state(self, data, latent_state, dts, odeint_rtol=None, odeint_atol=None, method=None):
        """
            predict the next latent state based on the input and the last latent state
            @:param data, shape [N, D]
                    latent_state, shape [N, D_latent]
                    dts, shape [N,]
            @:return shape [N, D_latent]
        """
        N = data.size(0)
        ts, inv_indices = torch.unique(dts, return_inverse=True)
        if ts[-1] == 0:
            return latent_state
        if ts[0] != 0:
            ts = torch.cat([torch.zeros(1, dtype=torch.float, device=self.device), ts])
            inv_indices += 1
        aug_latent_state = self.aug_layer(torch.cat((data, latent_state), dim=-1))
        traj_latent_state = self.diffeq_solver(aug_latent_state, ts, odeint_rtol, odeint_atol, method)
        selected_indices = tuple([torch.arange(N, dtype=torch.long, device=self.device), inv_indices])
        new_latent_state = traj_latent_state[selected_indices]  # [N, D_latent]
        assert new_latent_state.size(0) == N
        assert new_latent_state.size(1) == self.latent_dim
        return new_latent_state


class ActorODE(LatentODE):
    def __init__(self, state_dim, action_dim, input_dim, latent_dim, eps_decay,
                 encoder_z0, decoder, diffeq_solver, z0_prior, device, use_ode=True):
        super().__init__(
            input_dim, latent_dim, eps_decay, encoder_z0,
            decoder, diffeq_solver, None, z0_prior, device)
        self.num_states = state_dim
        self.num_actions = action_dim
        self.use_ode = use_ode
        a_input_dim = self.num_states
        if use_ode:
            a_input_dim += self.latent_dim
        self.actor_mlp = Actor(a_input_dim, self.num_actions,
            hidden1_dim=64, hidden2_dim=64).to(self.device)

    def forward(self, states, cur_hidden):
        if len(states.shape) == 1:
            states = states.unsqueeze(0)
        states = torch.cat([states, cur_hidden], dim=-1)
        return self.actor_mlp(states)


class CriticODE(LatentODE):
    def __init__(self, state_dim, action_dim, input_dim, latent_dim, eps_decay,
                 encoder_z0, decoder, diffeq_solver, z0_prior, device, use_ode=True):
        super().__init__(
            input_dim, latent_dim, eps_decay, encoder_z0,
            decoder, diffeq_solver, None, z0_prior, device)
        self.num_states = state_dim
        self.num_actions = action_dim
        self.use_ode = use_ode
        c_input_dim = self.num_states
        if use_ode:
            c_input_dim += self.latent_dim
        self.critic_mlp = Critic(c_input_dim, 1, self.num_actions,
            hidden1_dim=64, hidden2_dim=64).to(self.device)

    def forward(self, states, actions, cur_hidden):
        states = torch.cat([states, cur_hidden], dim=-1)
        return self.critic_mlp(states, actions)
