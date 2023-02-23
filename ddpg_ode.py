from policy import PolicyBase, soft_update, Actor, Critic
from model import ActorODE, CriticODE
import replay_memory
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

'''
states: s0      s1     s2
action:     a0      a1      a2
times : t0      t1     t2       t3

actor:  (s, a) from 0 to T --[ODE]--> h_{T+1} -[Linear]-> a_{T+1}
critic: (s, a) from 0 to T --[ODE]--> h_{T+1} -[Linear]-> V(s_{T+1})
'''


class Policy_ODE_DDPG(PolicyBase):
    def __init__(self, state_dim, action_dim, device, actor_lr=0.0001, critic_lr=0.001,
        batch_size=128, gamma=0.99, target_update=0.001, policy_actor:ActorODE=None,
        policy_critic:CriticODE=None, target_actor:ActorODE=None, target_critic:CriticODE=None,
        func_encode_action=lambda x: x):
         # conf
        super(Policy_ODE_DDPG, self).__init__(state_dim, action_dim, device, gamma=gamma, latent=False)
        self.batch_size = batch_size
        self.target_update = target_update
        self.func_encode_action = lambda x, y, z: torch.tensor(func_encode_action(x),
                                                               dtype=torch.float, device=self.device)
        self.input_dim = self.num_actions + self.num_actions
        # model
        self.policy_actor = policy_actor
        self.policy_critic = policy_critic
        self.target_actor = target_actor
        self.target_critic = target_critic
        # load model states, set to eval mode
        self.target_critic.load_state_dict(self.policy_critic.state_dict())
        self.target_actor.load_state_dict(self.policy_actor.state_dict())
        self.target_actor.eval()
        self.target_critic.eval()
        # optimizer
        self.optimizer_actor = optim.Adam(self.policy_actor.parameters(), lr=actor_lr)
        self.optimizer_critic = optim.Adam(self.policy_critic.parameters(), lr=critic_lr)
        self.criterion = nn.MSELoss()

    def __repr__(self):
        return "DDPG-ODE"

    def select_action(self, state, cur_hidden, eps=None):
        if type(state) is not torch.Tensor:
            state = torch.tensor(state, device=self.device, dtype=torch.float).unsqueeze(0)

        if eps is not None and np.random.uniform() < eps:
            action = np.random.uniform(-1, 1, size=self.num_actions)
        else:
            with torch.no_grad():
                action = self.policy_actor(state, cur_hidden).detach().cpu().numpy()
            action += self.noise
            action = np.clip(action, -1, 1)
        return action

    def select_action_in_batch(self, states, cur_hiddens, noise=True):
        with torch.no_grad():
            actions = self.policy_actor(states, cur_hiddens, train=False)
        if noise:
            actions += torch.empty_like(actions, dtype=torch.float, device=self.device).normal_(0, 0.1)
        actions = torch.clamp(actions, -1, 1)
        return actions

    def calc_value_in_batch(self, states, actions, cur_hiddens):
        return self.policy_critic(
            states, actions, cur_hiddens
        ).squeeze(-1)


    def optimize_mlp(self, memory, Trajectory, rms=None, train_ode=False):
        """
            Optimize DDPG
        """
        if len(memory) < self.batch_size:
            return
        trajs = memory.sample(self.batch_size)
        batch: replay_memory.Trajectory = Trajectory(*zip(*trajs))
        state_traj_batch = torch.stack(batch.states)
        action_traj_batch = torch.stack(batch.actions)
        timestep_batch = torch.stack(batch.time_steps) # [B, T+1]

        state_batch = state_traj_batch[:, -2, :]  # [B, D_state]
        next_state_batch = state_traj_batch[:, -1, :]  # [B, D_state]

        if rms is not None:
            state_batch = rms.normalize(state_batch)
            next_state_batch = rms.normalize(next_state_batch)

        
        action_batch = action_traj_batch[:, -1, :]  # [B, D_action]
        reward_batch = torch.stack(batch.rewards)[:, -1]  # [B,]
        dt_batch = timestep_batch[:, -1] - timestep_batch[:, -2]  # [B,]

        if not train_ode:
            actor_latent_traj_batch = torch.stack(batch.actor_hiddens).detach()  # [B, T+1, D_latent]
            critic_latent_traj_batch = torch.stack(batch.critic_hiddens).detach() # [B, T+1, D_latent]
            actor_latent_batch = actor_latent_traj_batch[:, -2]
            next_actor_latent_batch = actor_latent_traj_batch[:, -1]
            critic_latent_batch = critic_latent_traj_batch[:, -2]
            next_critic_latent_batch = critic_latent_traj_batch[:, -1]
        else:
            hs_actor, _, _ = self.policy_actor.encode_latent_traj(state_traj_batch[:, :-1, :], action_traj_batch, timestep_batch, 
                torch.tensor(batch.length, device=self.device), True)
            hs_critic, _, _ = self.policy_critic.encode_latent_traj(state_traj_batch[:, :-1, :], action_traj_batch, timestep_batch, 
                torch.tensor(batch.length, device=self.device), True)
            actor_latent_batch = hs_actor[:, -2]
            next_actor_latent_batch = hs_actor[:, -1]
            critic_latent_batch = hs_critic[:, -2]
            next_critic_latent_batch = hs_critic[:, -1]

        assert state_batch.size(0) == self.batch_size
        assert len(reward_batch.size()) == 1

        # compute Q(s_t, a)
        state_action_values = self.policy_critic(
            state_batch, action_batch, critic_latent_batch).squeeze(1)  # [B,]

        # compute max_a Q(s_{t+1}, a) for all next states
        next_action_batch = self.target_actor(
            next_state_batch, next_actor_latent_batch
        ).detach()
        next_state_values = self.target_critic(
            next_state_batch, next_action_batch, next_critic_latent_batch
        ).squeeze(1).detach()

        # compute the expected Q values
        expected_state_action_values = reward_batch + (self.gamma ** dt_batch) * next_state_values

        # compute critic loss
        critic_loss = self.criterion(state_action_values, expected_state_action_values)

        # optimize the critic
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # compute actor loss
        actor_loss = -self.policy_critic(
            state_batch, 
            self.policy_actor(state_batch, actor_latent_batch),
            critic_latent_batch
        ).mean()

        # optimize the actor
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # update target
        soft_update(self.target_actor, self.policy_actor, self.target_update)
        soft_update(self.target_critic, self.policy_critic, self.target_update)

        return critic_loss.item(), actor_loss.item()


    def optimize(self, memory, Transition, rms=None):
        """
            Optimize DDPG
            (s, a, s', r, dt, h)
        """
        if len(memory) < self.batch_size:
            return
        t0 = time.time()
        transitions = memory.sample(self.batch_size)
        batch: replay_memory.Transition = Transition(*zip(*transitions))

        ### Get batch of normalized states, non-final next states and state trajs
        state_batch = torch.stack(batch.state)  # [B, D_state]
        state_traj_batch =  torch.stack(batch.state_traj)  # [B, T, D_state]
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_state_batch = torch.stack([s for s in batch.next_state if s is not None])  # [B, D_state]
        if rms is not None:
            state_batch = rms.normalize(state_batch)
            non_final_next_state_batch = rms.normalize(non_final_next_state_batch)
            state_traj_batch = rms.normalize(state_traj_batch)

        ### Get batch of action, action_traj, dt, ts_traj, and reward
        action_batch = torch.stack([self.func_encode_action(a, self.num_actions, self.device)
                                    for a in batch.action])  # [B, D_action]
        action_traj_batch = torch.stack(batch.action_traj)  # [B, T, D_action]
        dt_batch = torch.tensor(batch.dt, dtype=torch.float, device=self.device)  # [B,]
        ts_traj_batch = torch.stack(batch.ts_traj)  # [B, T+1]
        reward_batch = torch.tensor(batch.reward, dtype=torch.float, device=self.device)  # [B,]

        print("traj shapes: {}, {}, {}".format(state_traj_batch.shape, action_traj_batch.shape, ts_traj_batch.shape))

        ### non-final trajs: time t_{1:n+1} ###
        s_trajs = torch.cat([state_traj_batch, state_batch.unsqueeze(1)], dim=1)  # [B, T+1, D_state]
        a_trajs = torch.cat([action_traj_batch, action_batch.unsqueeze(1)], dim=1)  # [B, T+1, D_action]
        t_trajs = torch.cat([ts_traj_batch, (dt_batch + ts_traj_batch[:, -1]).unsqueeze(1)], dim=1)  # [B, T+2]
        non_final_next_state_traj_batch = torch.stack([s_trajs[i] 
            for i in range(len(s_trajs)) if batch.next_state[i] is not None])
        non_final_next_action_traj_batch = torch.stack([a_trajs[i] 
            for i in range(len(a_trajs)) if batch.next_state[i] is not None])
        non_final_next_ts_traj_batch = torch.stack([t_trajs[i] 
            for i in range(len(t_trajs)) if batch.next_state[i] is not None])

        assert state_batch.size(0) == self.batch_size
        assert len(reward_batch.size()) == 1

        # compute Q(s_t, a)
        state_action_values = self.policy_critic(state_batch, action_batch,
            state_traj_batch, action_traj_batch, ts_traj_batch, train=True)[0].squeeze(1)

        # compute max_a Q(s_{t+1}, a) for all next states
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        # 1. compute a(s_{t+1})
        next_action_batch, next_hidden_batch = self.target_actor(
            non_final_next_state_batch, non_final_next_state_traj_batch, 
            non_final_next_action_traj_batch, non_final_next_ts_traj_batch, train=True
        ).detach()
        # 2. compute Q(s_{t+1}, a_(s_{t+1}))
        qvals, hvals = self.target_critic(
            non_final_next_state_batch, next_action_batch, non_final_next_state_traj_batch, 
            non_final_next_action_traj_batch, non_final_next_ts_traj_batch, train=True
        ).squeeze(1).detach()
        # 3. assign qvals to the non-final cells
        next_state_values[non_final_mask] = qvals
        # compute the expected Q values
        expected_state_action_values = reward_batch + (self.gamma ** dt_batch) * next_state_values

        # compute critic loss
        critic_loss = self.criterion(state_action_values, expected_state_action_values)

        # optimize the critic
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # compute actor loss
        a_temp, h_temp = self.policy_actor(state_batch, state_traj_batch, 
            action_traj_batch, ts_traj_batch, train=True)
        actor_loss = -self.policy_critic(state_batch, a_temp, state_traj_batch, 
            action_traj_batch, ts_traj_batch, train=True)[0].mean()

        # optimize the actor
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # update target
        soft_update(self.target_actor, self.policy_actor, self.target_update)
        soft_update(self.target_critic, self.policy_critic, self.target_update)
        # print("loss:", actor_loss, critic_loss)
        t1 = time.time()
        print("time:", t1-t0)
        return critic_loss.item(), actor_loss.item()


    @property
    def noise(self):
        return np.random.normal(0, 0.1, size=self.num_actions)
