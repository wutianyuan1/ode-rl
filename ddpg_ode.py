from policy import PolicyBase, Critic, soft_update
from model import ModelFreeODE, MLP
import replay_memory
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
states: s0      s1     s2
action:     a0      a1      a2
times : t0      t1     t2       t3

actor:  (s, a) from 0 to T --[ODE]--> h_{T+1} -[Linear]-> a_{T+1}
critic: (s, a) from 0 to T --[ODE]--> h_{T+1} -[Linear]-> V(s_{T+1})
'''


class Policy_ODE_DDPG(PolicyBase):
    def __init__(self, state_dim, action_dim, device, actor_lr=0.0001, critic_lr=0.001,
        batch_size=128, gamma=0.99, target_update=0.001, policy_model:ModelFreeODE=None,
        target_model:ModelFreeODE=None, func_encode_action=lambda x: x):
         # conf
        super(Policy_ODE_DDPG, self).__init__(state_dim, action_dim, device, gamma=gamma, latent=False)
        self.batch_size = batch_size
        self.target_update = target_update
        self.func_encode_action = lambda x, y, z: torch.tensor(func_encode_action(x),
                                                               dtype=torch.float, device=self.device)
        self.input_dim = self.num_actions + self.num_actions
        # model
        self.policy_ode_model = policy_model
        self.target_ode_model = target_model
        # load model states
        self.target_ode_model.load_state_dict(self.policy_ode_model.state_dict())
        self.target_ode_model.eval()
        # optimizer
        self.optimizer = optim.Adam(self.policy_ode_model.parameters(), lr=actor_lr)
        self.criterion = nn.MSELoss()

    def __repr__(self):
        return "DDPG-ODE"

    def select_action(self, state, state_traj, action_traj, ts_traj, eps=None):
        """
            Return actor(s)
            state: [D_state, ]
            state_traj: [len, D_state]
            action_traj: [len, D_action]
            ts_traj: [len+1, ]
        """
        if type(state) is not torch.Tensor:
            state = torch.tensor(state, device=self.device, dtype=torch.float).unsqueeze(0)
        if type(state_traj) is not torch.Tensor:
            state_traj = torch.tensor(state_traj, device=self.device, dtype=torch.float).unsqueeze(0)
        if type(action_traj) is not torch.Tensor:
            action_traj = torch.tensor(action_traj, device=self.device, dtype=torch.float).unsqueeze(0)
        if type(ts_traj) is not torch.Tensor:
            ts_traj = torch.tensor(ts_traj, device=self.device, dtype=torch.float).unsqueeze(0)

        if eps is not None and np.random.uniform() < eps:
            action = np.random.uniform(-1, 1, size=self.num_actions)
        else:
            action = self.policy_ode_model.compute_action(
                state, state_traj, action_traj, ts_traj).cpu().numpy()
            print(action)
            action += self.noise
            action = np.clip(action, -1, 1)
        return action

    def select_action_in_batch(self, states, state_trajs, action_trajs, ts_trajs, noise=True):
        actions = self.policy_ode.compute_action(states, state_trajs, action_trajs, ts_trajs)
        if noise:
            actions += torch.empty_like(actions, dtype=torch.float, device=self.device).normal_(0, 0.1)
        actions = torch.clamp(actions, -1, 1)
        return actions

    def calc_value_in_batch(self, states, actions, state_trajs, action_trajs, ts_trajs):
        return self.policy_ode_model.compute_value(
            states, actions, state_trajs, action_trajs, ts_trajs
        ).squeeze(-1)

    def optimize(self, memory, Transition, rms=None):
        """
            Optimize DDPG
            (s, a, s', r, dt, h)
        """
        if len(memory) < self.batch_size:
            return
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
        state_action_values = self.policy_ode_model.compute_value(state_batch, action_batch,
            state_traj_batch, action_traj_batch, ts_traj_batch, train=True).squeeze(1)

        # compute max_a Q(s_{t+1}, a) for all next states
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        # 1. compute a(s_{t+1})
        next_action_batch = self.target_ode_model.compute_action(
            non_final_next_state_batch, non_final_next_state_traj_batch, 
            non_final_next_action_traj_batch, non_final_next_ts_traj_batch, train=True
        ).detach()
        # 2. compute Q(s_{t+1}, a_(s_{t+1}))
        qvals = self.target_ode_model.compute_value(
            non_final_next_state_batch, next_action_batch, non_final_next_state_traj_batch, 
            non_final_next_action_traj_batch, non_final_next_ts_traj_batch, train=True
        ).squeeze(1).detach()
        # 3. assign qvals to the non-final cells
        next_state_values[non_final_mask] = qvals
        # compute the expected Q values
        expected_state_action_values = reward_batch + (self.gamma ** dt_batch) * next_state_values

        # compute critic loss
        # print("state diff:", state_action_values, expected_state_action_values)
        critic_loss = self.criterion(state_action_values, expected_state_action_values)
        # print("critic loss:", critic_loss)

        # optimize the critic
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        # compute actor loss
        a_temp = self.policy_ode_model.compute_action(state_batch, state_traj_batch, 
            action_traj_batch, ts_traj_batch, train=True)
        actor_loss = -self.policy_ode_model.compute_value(state_batch, a_temp, state_traj_batch, 
            action_traj_batch, ts_traj_batch, train=True).mean()

        # optimize the actor
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        # update target
        soft_update(self.target_ode_model, self.policy_ode_model, self.target_update)
        # print("loss:", actor_loss, critic_loss)
        return critic_loss.item(), actor_loss.item()



    @property
    def noise(self):
        return np.random.normal(0, 0.1, size=self.num_actions)

