#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint
import yaml
import sys

import numpy as np
import torch
from envs.make_env import make_env
from torch import nn
from torch.distributions import Independent, Normal
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from model import RecurrentActorProb, RecurrentCritic
from ode_model import NeuralCDEActorProb, NeuralCDECritic, MLPActor
from policy import PPOPolicy

class Runner(object):
    @staticmethod
    def dist(*logits):
        loc, scale = logits
        return Independent(Normal(*logits), 1)

    def __init__(self, config_path) -> None:
        self.config_path = config_path
        with open(self.config_path, 'r') as f:
            conf = yaml.safe_load(f)
        # Add config attributes to self
        for key, value in conf.items():
            self.__dict__[key] = value
        print(self.__dict__)
        self.conf = conf
        self.env, self.train_envs, self.test_envs = self.construct_env()
        self.actor, self.critic = self.construct_net(self.net_type)
        self.initialize_params()
        self.policy = PPOPolicy(
            self.actor,
            self.critic,
            self.optim,
            self.dist,
            state_with_time=self.state_with_time,
            discount_factor=self.gamma,
            gae_lambda=self.gae_lambda,
            max_grad_norm=self.max_grad_norm,
            vf_coef=self.vf_coef,
            ent_coef=self.ent_coef,
            reward_normalization=self.rew_norm,
            action_scaling=True,
            action_bound_method=self.bound_action_method,
            lr_scheduler=self.lr_scheduler,
            action_space=self.env.action_space,
            eps_clip=self.eps_clip,
            value_clip=self.value_clip,
            dual_clip=self.dual_clip,
            advantage_normalization=self.norm_adv,
            recompute_advantage=self.recompute_adv,
        )
        self.resume()
        self.train_collector, self.test_collector = self.construct_collecter(
            stack_num=1 if self.net_type == 'MLP' else self.history_len)
        self.construct_logger()
        
    def construct_net(self, net_type):
        if net_type == 'MLP':
            net_a = Net(
                self.state_shape,
                hidden_sizes=self.hidden_sizes,
                activation=nn.Tanh,
                device=self.device,
            )
            actor = ActorProb(
                net_a,
                self.action_shape,
                max_action=self.max_action,
                unbounded=True,
                device=self.device,
            ).to(self.device)
            net_c = Net(
                self.state_shape,
                hidden_sizes=self.hidden_sizes,
                activation=nn.Tanh,
                device=self.device,
            )
            critic = Critic(net_c, device=self.device).to(self.device)
        elif net_type == 'RNN':
            actor = RecurrentActorProb(
                layer_num=3,
                state_shape=self.state_shape,
                action_shape=self.action_shape,
                max_action=self.max_action,
                device=self.device,
                hidden_layer_size=int(np.mean(self.hidden_sizes)),
                unbounded=True,
            ).to(self.device)
            critic = RecurrentCritic(
                layer_num=3,
                state_shape=self.state_shape,
                action_shape=self.action_shape,
                device=self.device,
                hidden_layer_size=int(np.mean(self.hidden_sizes)),
                target='v'
            ).to(self.device)
        elif net_type == 'CDE':
            if not self.both_ode:
                net_a = Net(
                    self.state_shape,
                    hidden_sizes=self.hidden_sizes,
                    activation=nn.Tanh,
                    device=self.device,
                )
                actor = MLPActor(
                    net_a,
                    self.action_shape,
                    max_action=self.max_action,
                    unbounded=True,
                    device=self.device,
                ).to(self.device)
            else:
                actor = NeuralCDEActorProb(
                    layer_num=3,
                    state_shape=self.state_shape,
                    action_shape=self.action_shape,
                    max_action=self.max_action,
                    device=self.device,
                    hidden_layer_size=int(np.mean(self.hidden_sizes)),
                    unbounded=False,
                ).to(self.device)
            critic = NeuralCDECritic(
                layer_num=3,
                state_shape=self.state_shape,
                action_shape=self.action_shape,
                device=self.device,
                hidden_layer_size=int(np.mean(self.hidden_sizes)),
                target='v'
            ).to(self.device)
        else:
            raise NotImplementedError("Unimplemented network: " + net_type)
        return actor, critic
  
    def construct_env(self):
        env, train_envs, test_envs = make_env(
            self.task, self.seed, self.training_num, self.test_num,
            obs_norm=True, state_with_time=self.state_with_time
        )
        self.state_shape = env.observation_space.shape or env.observation_space.n
        self.action_shape = env.action_space.shape or env.action_space.n
        self.max_action = env.action_space.high[0]
        print("Observations shape:", self.state_shape)
        print("Actions shape:", self.action_shape)
        print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))
        # seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        return env, train_envs, test_envs

    def construct_collecter(self, stack_num):
        # collector
        if self.training_num > 1:
            buffer = VectorReplayBuffer(self.buffer_size, len(self.train_envs), stack_num=stack_num)
        else:
            buffer = ReplayBuffer(self.buffer_size, stack_num=stack_num)
        train_collector = Collector(self.policy, self.train_envs, buffer, exploration_noise=True)
        test_collector = Collector(self.policy, self.test_envs)
        return train_collector, test_collector

    def construct_logger(self):
        now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        self.algo_name = "ppo"
        log_name = os.path.join(self.task, self.algo_name, str(self.seed), now)
        self.log_path = os.path.join(self.logdir, log_name)
        # logger
        if self.logger == "wandb":
            self.logger = WandbLogger(
                save_interval=1,
                name=log_name.replace(os.path.sep, "__"),
                run_id=self.resume_id,
                config=self,
                project=self.wandb_project,
            )
        writer = SummaryWriter(self.log_path)
        writer.add_text("args", str(self))
        if self.logger == "tensorboard":
            self.logger = TensorboardLogger(writer)
        else:  # wandb
            self.logger.load(writer)
        with open(self.log_path + '/config.yml', 'w') as conf:
            conf.write(str(self.conf))

    def initialize_params(self):
        torch.nn.init.constant_(self.actor.sigma_param, -0.5)
        for m in list(self.actor.modules()) + list(self.critic.modules()):
            if isinstance(m, torch.nn.Linear):
                # orthogonal initialization
                torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                torch.nn.init.zeros_(m.bias)
        # do last policy layer scaling, this will make initial actions have (close to)
        # 0 mean and std, and will help boost performances,
        # see https://arxiv.org/abs/2006.05990, Fig.24 for details
        for m in self.actor.mu.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.bias)
                m.weight.data.copy_(0.01 * m.weight.data)

        self.optim = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.lr
        )

        self.lr_scheduler = None
        if self.lr_decay:
            # decay learning rate to 0 linearly
            max_update_num = np.ceil(
                self.step_per_epoch / self.step_per_collect
            ) * self.epoch

            self.lr_scheduler = LambdaLR(
                self.optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num
            )

    def resume(self):
        # load a previous policy
        if self.resume_path:
            ckpt = torch.load(self.resume_path, map_location=self.device)
            self.policy.load_state_dict(ckpt["model"])
            self.train_envs.set_obs_rms(ckpt["obs_rms"])
            self.test_envs.set_obs_rms(ckpt["obs_rms"])
            print("Loaded agent from: ", self.resume_path)
    
    def save_best_fn(self, policy):
        state = {"model": policy.state_dict(), "obs_rms": self.train_envs.get_obs_rms()}
        torch.save(state, os.path.join(self.log_path, "policy.pth"))

    def train(self):
        result = onpolicy_trainer(
            self.policy,
            self.train_collector,
            self.test_collector,
            self.epoch,
            self.step_per_epoch,
            self.repeat_per_collect,
            self.test_num,
            self.batch_size,
            step_per_collect=self.step_per_collect,
            save_best_fn=self.save_best_fn,
            logger=self.logger,
            test_in_train=False,
        )
        pprint.pprint(result)
        return result

    def eval(self):
        # Let's watch its performance!
        self.policy.eval()
        self.test_envs.seed(self.seed)
        self.test_collector.reset()
        result = self.test_collector.collect(n_episode=self.test_num, render=self.render)
        print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')
        return result


if __name__ == "__main__":
    runner = Runner(sys.argv[1])
    runner.train()
    runner.eval()
    # test_ppo()