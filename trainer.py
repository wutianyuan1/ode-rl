#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint
import yaml
import sys

import numpy as np
import torch
import torch.nn as nn
from envs.make_env import make_env
from torch import nn
import torch.optim as optim
from torch.distributions import Independent, Normal
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from models.rnn_model import GRUActorProb, GRUCritic, LSTMActorProb, LSTMCritic
from models.cde_model import NeuralCDEActorProb, NeuralCDECritic
from models.hbode_model import HBOdeCritic
from models.s4_model import S4ActorProb, S4Critic
from models.mlp_model import MLPActorProb, MLPCritic
from models.representation import RepresentationMLP
from ppo import PPOPolicy


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
        # print(self.__dict__)
        self.conf = conf
        self.env, self.train_envs, self.test_envs = self.construct_env()
        self.actor, self.critic = self.construct_net(self.actor_type, self.critic_type)
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
        no_recurrence = (self.actor_type == 'MLP' and self.critic_type == 'MLP')
        self.train_collector, self.test_collector = self.construct_collecter(
            stack_num=1 if no_recurrence else self.history_len)
        self.construct_logger(self.expr_id)
        
    def construct_net(self, actor_type, critic_type):
        print("state:", self.state_shape, "hiddens:", self.hidden_sizes)
        preprocess = RepresentationMLP(np.prod(self.state_shape), self.hidden_sizes, nn.Tanh, device=self.device).to(self.device)
        ## Construct Actor !!
        if actor_type == 'MLP':
            actor = MLPActorProb(
                preprocess,
                self.action_shape,
                max_action=self.max_action,
                unbounded=True,
                device=self.device,
            ).to(self.device)
        elif actor_type == 'GRU':
            actor = GRUActorProb(
                preprocess, 4, self.state_shape, self.action_shape,
                max_action=self.max_action,
                device=self.device,
                hidden_layer_size=self.hidden_sizes[-1],
                unbounded=True,
            ).to(self.device)
        elif actor_type == 'LSTM':
            actor = LSTMActorProb(
                preprocess, 4, self.state_shape, self.action_shape,
                max_action=self.max_action,
                device=self.device,
                hidden_layer_size=self.hidden_sizes[-1],
                unbounded=True,
            ).to(self.device)
        elif actor_type == 'S4':
            actor = S4ActorProb(
                preprocess, 4, self.state_shape, self.action_shape,
                max_action=self.max_action,
                device=self.device,
                hidden_layer_size=self.hidden_sizes[-1],
                unbounded=True,
            ).to(self.device)
        elif actor_type == 'CDE':
            actor = NeuralCDEActorProb(
                preprocess, 4, self.state_shape, self.action_shape,
                max_action=self.max_action,
                device=self.device,
                hidden_layer_size=self.hidden_sizes[-1],
                unbounded=True,
            ).to(self.device)
        else:
            raise NotImplementedError("Unimplemented actor: " + actor_type)

        ## Construct Critic !!
        if critic_type == 'MLP':
            critic = MLPCritic(preprocess, device=self.device).to(self.device)
        elif critic_type == 'S4':
            critic = S4Critic(preprocess, 4, self.state_shape, self.action_shape, self.device, self.hidden_sizes[-1]).to(self.device)
        elif critic_type == 'GRU':
            critic = GRUCritic(preprocess, 4, self.state_shape, self.action_shape, self.device, self.hidden_sizes[-1]).to(self.device)
        elif critic_type == 'LSTM':
            critic = LSTMCritic(preprocess, 4, self.state_shape, self.action_shape, self.device, self.hidden_sizes[-1]).to(self.device)
        elif critic_type == 'CDE':
            critic = NeuralCDECritic(preprocess, 4, self.state_shape, self.action_shape, self.device, self.hidden_sizes[-1]).to(self.device)
        elif critic_type == 'HBODE':
            critic = HBOdeCritic(preprocess, 4, self.state_shape, self.action_shape, self.device, self.hidden_sizes[-1]).to(self.device)
        else:
            raise NotImplementedError("Unimplemented critic: " + actor_type)
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

    def construct_logger(self, expr_id):
        now = datetime.datetime.now().strftime("%y%m%d-%H%M%S") + str(expr_id)
        self.algo_name = "ppo"
        log_name = os.path.join(self.task, self.algo_name, str(self.seed), now)
        self.log_path = os.path.join(self.logdir, log_name)
        # logger
        writer = SummaryWriter(self.log_path)
        writer.add_text("args", str(self))
        if self.logger == "tensorboard":
            self.logger = TensorboardLogger(writer)
        else:  # wandb
            self.logger.load(writer)
        with open(self.log_path + '/config.yml', 'w') as conf:
            conf.write(str(self.conf))

    def setup_optimizer(self, actor, critic):
        # All parameters in the model
        all_parameters = list(actor.parameters())
        for (name, param) in critic.named_parameters():
            if 'preprocess' not in name:
                all_parameters.append(param)

        # General parameters don't contain the special _optim key
        params = [p for p in all_parameters if not hasattr(p, "_optim")]

        # Create an optimizer with the general parameters
        optimizer = optim.AdamW(params, lr=self.lr, weight_decay=0.01)

        # Add parameters with special hyperparameters
        hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
        hps = [
            dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
        ]  # Unique dicts
        for hp in hps:
            params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group(
                {"params": params, **hp}
            )
        # Print optimizer info
        keys = sorted(set([k for hp in hps for k in hp.keys()]))
        for i, g in enumerate(optimizer.param_groups):
            group_hps = {k: g.get(k, None) for k in keys}
            print(' | '.join([
                f"Optimizer group {i}",
                f"{len(g['params'])} tensors",
            ] + [f"{k} {v}" for k, v in group_hps.items()]))
        return optimizer

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

        self.optim = self.setup_optimizer(self.actor, self.critic)

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
