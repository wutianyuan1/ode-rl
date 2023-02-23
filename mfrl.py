import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.optim as optim
from ddpg_ode import Policy_ODE_DDPG
from envs.half_cheetah_simulator import HalfCheetahSimulator
from model import Encoder_z0_RNN, DiffeqSolver, ODEFunc, ActorODE, CriticODE
from policy import Actor, Critic
from replay_memory import ReplayMemory, Trajectory
from running_stats import RunningStats
from tqdm import trange
import utils


class ModelFreeODERL(object):
    def __init__(self, simulator, gamma=0.99, mem_size=int(1e5), lr=9e-4, batch_size=32, ode_tol=1e-3, ode_dim=20,
                 enc_hidden_to_latent_dim=20, latent_dim=10, eps_decay=1e-4, weight_decay=1e-3, obs_normal=False,
                 exp_id=0, trained_model_path='', ckpt_path='', seed=2023, logger=None, actor_use_ode=True,
                 critic_use_ode=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exp_id = exp_id
        self.simulator: HalfCheetahSimulator = simulator
        self.batch_size = batch_size
        self.input_dim = self.simulator.num_states + self.simulator.num_actions
        self.output_dim = self.simulator.num_states
        self.latent_dim = latent_dim
        self.ckpt_path = ckpt_path
        self.logger = logger
        self.rms = RunningStats(dim=self.simulator.num_states, device=self.device) if obs_normal else None
        self.seed = seed
        self.actor_use_ode = actor_use_ode
        self.critic_use_ode = critic_use_ode

        self.policy_actor = self.create_actor_critic("actor", self.actor_use_ode, latent_dim, 
            enc_hidden_to_latent_dim, eps_decay, ode_dim, ode_tol)
        self.policy_critic = self.create_actor_critic("critic", self.critic_use_ode, latent_dim, 
            enc_hidden_to_latent_dim, eps_decay, ode_dim, ode_tol)
        self.target_actor = self.create_actor_critic("actor", self.actor_use_ode, latent_dim, 
            enc_hidden_to_latent_dim, eps_decay, ode_dim, ode_tol)
        self.target_critic = self.create_actor_critic("critic", self.critic_use_ode, latent_dim, 
            enc_hidden_to_latent_dim, eps_decay, ode_dim, ode_tol)

        # policy and replay buffer
        self.policy = Policy_ODE_DDPG(
            state_dim=self.simulator.num_states, action_dim=self.simulator.num_actions,
            device=self.device, gamma=gamma,
            batch_size=batch_size,
            policy_actor=self.policy_actor, policy_critic=self.policy_critic,
            target_actor=self.target_actor, target_critic=self.target_critic
        )

        self.memory_trajs = ReplayMemory(mem_size, Trajectory)

        if trained_model_path:
            self.model.load_state_dict(torch.load(trained_model_path, map_location=self.device)['model_state_dict'])
            self.target_model.load_state_dict(torch.load(trained_model_path, map_location=self.device)['model_state_dict'])


    def create_actor_critic(self, ac_type, use_ode, latent_dim, enc_hidden_to_latent_dim, eps_decay, ode_dim, ode_tol):
        if not use_ode:
            if ac_type == 'actor':
                return Actor(self.simulator.num_states, self.simulator.num_actions,
                             hidden1_dim=64, hidden2_dim=64).to(self.device)
            else:
                return Critic(self.simulator.num_states, 1, self.simulator.num_actions,
                              hidden1_dim=64, hidden2_dim=64).to(self.device)

        # ode network   
        gen_ode_func = ODEFunc(ode_func_net=utils.create_net(latent_dim, latent_dim, n_layers=2, n_units=ode_dim,
                                                                nonlinear=nn.Tanh)).to(self.device)
        diffq_solver = DiffeqSolver(gen_ode_func, 'dopri5', odeint_rtol=ode_tol, odeint_atol=ode_tol/10)
        # encoder
        encoder = Encoder_z0_RNN(latent_dim, self.simulator.num_states, hidden_to_z0_units=enc_hidden_to_latent_dim,
                                    device=self.device).to(self.device)
        z0_prior = Normal(torch.tensor([0.]).to(self.device), torch.tensor([1.]).to(self.device))
        if ac_type == 'actor':
            return ActorODE(
                state_dim=self.simulator.num_states,
                action_dim=self.simulator.num_actions,
                input_dim=self.simulator.num_states,
                latent_dim=latent_dim,
                eps_decay=eps_decay,
                encoder_z0=encoder,
                decoder=None,
                diffeq_solver=diffq_solver,
                z0_prior=z0_prior,
                device=self.device,
                use_ode=use_ode).to(self.device)
        else:
            return CriticODE(
                state_dim=self.simulator.num_states,
                action_dim=self.simulator.num_actions,
                input_dim=self.simulator.num_states,
                latent_dim=self.latent_dim,
                eps_decay=eps_decay,
                encoder_z0=encoder,
                decoder=None,
                diffeq_solver=diffq_solver,
                z0_prior=z0_prior,
                device=self.device,
                use_ode=use_ode
            ).to(self.device)


    def run_policy(self, max_steps, eps=None, store_trans=True):
        """
            Run policy once with interaction with environment, optimize policy and save transitions and trajectories
            Note that when we call this function, env model should been trained to generate reasonable latent states
        """
        states = [torch.tensor(self.simulator.reset(seed=np.random.randint(0,2023))[0], dtype=torch.float, device=self.device)]
        actions_encoded, rewards, dts = [], [], [0.]
        actor_hiddens  =  [self.policy_actor.sample_init_latent_states().unsqueeze(0)]
        critic_hiddens =  [self.policy_critic.sample_init_latent_states().unsqueeze(0)]
        history_len = 10  ##### HISTORY LENGTH

        for i_step in trange(max_steps):
            state = states[-1]
            a_hidden = actor_hiddens[-1]
            c_hidden = critic_hiddens[-1]
            
            if self.rms is not None:
                self.rms += state
            norm_state = state if self.rms is None else self.rms.normalize(state)

            action = self.policy.select_action(norm_state, a_hidden, eps=eps)
            if self.actor_use_ode:
                action = action[0]
            action_encoded = torch.tensor(action, device=self.device).unsqueeze(0)
            dt = self.simulator.get_time_gap(action=action)
            next_state, reward, done, info = self.simulator.step(action, dt=dt)
            next_state = torch.tensor(next_state, dtype=torch.float, device=self.device)
            
            ## Update buffers
            actor_hiddens.append(self.policy_actor.encode_next_latent_state(
                state.unsqueeze(0), a_hidden, 
                torch.tensor([dt], device=self.device, dtype=torch.float)))
            critic_hiddens.append(self.policy_critic.encode_next_latent_state(
                state.unsqueeze(0), c_hidden, 
                torch.tensor([dt], device=self.device, dtype=torch.float)))

            states.append(next_state)
            actions_encoded.append(action_encoded)
            rewards.append(reward)
            dts.append(dt)

            ## Train the model
            self.policy.optimize_mlp(self.memory_trajs, Trajectory, self.rms, train_ode=True)
            # if i_step % 64 == 0:
            #     self.policy.optimize_ode(self.memory_trans, Trajectory, self.rms)

            if done:
                break

        ### !! store trajectory !!
        states = torch.stack(states)  # [T+1, D_state]
        actions_encoded = torch.stack(actions_encoded).squeeze(1)  # [T, D_action]
        time_steps = torch.tensor(dts, device=self.device, dtype=torch.float).cumsum(dim=0)  # [T+1, ]
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)  # [T,]
        actor_hiddens = torch.stack(actor_hiddens).squeeze(1)  # [T+1,D_latent]
        critic_hiddens = torch.stack(critic_hiddens).squeeze(1)  # [T+1,D_latent]
        self.save_traj_to_buffer(states, actions_encoded, time_steps, rewards, actor_hiddens, 
            critic_hiddens, max_steps, history_len)

        # calculate accumulated rewards
        acc_rewards = self.calc_acc_rewards(rewards, time_steps[:-1],
                                            discount=bool('HIV' in repr(self.simulator))).item()
        return acc_rewards

    def save_traj_to_buffer(self, states, actions_encoded, time_steps, rewards,
                            actor_hiddens, critic_hiddens, traj_len, unit):
        idx = 0
        T = actions_encoded.size(0)
        assert T != 0 and T % unit == 0
        while idx < traj_len:
            self.memory_trajs.push(states[idx:idx+unit+1], actions_encoded[idx:idx+unit],
                        time_steps[idx:idx+unit+1], rewards[idx:idx+unit], actor_hiddens[idx:idx+unit+1],
                        critic_hiddens[idx:idx+unit+1], unit if idx+unit < traj_len else traj_len-idx)
            idx += unit


    def calc_acc_rewards(self, rewards, time_steps, discount=False):
        """
            Calculate accumulated return base on semi-mdp
        """
        discounts = self.policy.gamma ** time_steps if discount \
            else torch.ones_like(time_steps, dtype=torch.float, device=self.device)
        if len(rewards.size()) == 1:  # [T,]
            return torch.dot(rewards, discounts)
        elif len(rewards.size()) == 2:  # [N, T]
            return torch.mm(rewards, discounts.t()).diag()
        else:
            raise ValueError("rewards should be 1D vector or 2D matrix.")


    def train(self, steps, train_episodes, cur_epoch, store_trans=True):
        t = time.time()
        rewards = []
        print("Train episodes: {}, steps: {}".format(train_episodes, steps))
        for episode_i in range(train_episodes):
            t1 = time.time()
            reward = self.run_policy(steps, store_trans=store_trans)
            t2 = time.time()

            rewards.append(reward)
            log = "Episode {} | Time = {} | reward = {:.6f}".format(episode_i, t2 - t1, reward)
            utils.logout(self.logger, log)

        eval_reward = 0
        for _ in range(1):
            eval_reward += self.run_policy(steps, store_trans=False)
        log = "Epoch {} | avg reward over last epoch = {:.6f} | eval reward = {:.6f}" \
              " | time = {:.6f} s".format(cur_epoch, sum(rewards) / len(rewards),
                                          eval_reward / 1.0, time.time() - t)
        utils.logout(self.logger, log)
        return rewards, eval_reward

