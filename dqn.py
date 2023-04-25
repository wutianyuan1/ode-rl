import datetime
import os
import pprint
import yaml
import sys

import numpy as np
import torch
from envs.make_env import make_env
from torch.utils.tensorboard import SummaryWriter
from ode_model import NerualCDEDQN, MLPDQN
from tianshou.data import Collector, PrioritizedVectorReplayBuffer
from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger


class Runner(object):
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
    
    def construct_env(self):
        env, train_envs, test_envs = make_env(
            self.task, self.seed, self.training_num, self.test_num,
            obs_norm=True, state_with_time=self.state_with_time
        )
        self.state_shape = env.observation_space.shape or env.observation_space.n
        self.action_shape = env.action_space.shape or env.action_space.n
        print("Observations shape:", self.state_shape)
        print("Actions shape:", self.action_shape)
        # seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        return env, train_envs, test_envs

    def test_dqn(self):
        print(self.model)
        if self.model == 'CDE':
            net = NerualCDEDQN(self.state_shape, self.action_shape, self.cde_hidden_size, self.device).to(self.device)
        elif self.model == 'MLP':
            net = MLPDQN(self.state_shape, self.action_shape, self.mlp_hidden_size, self.device).to(self.device)
        optim = torch.optim.Adam(net.parameters(), lr=self.lr)
        # define policy
        policy = DQNPolicy(
            net,
            optim,
            self.gamma,
            self.n_step,
            target_update_freq=self.target_update_freq
        )

        # load a previous policy
        if self.resume_path:
            policy.load_state_dict(torch.load(self.resume_path, map_location=self.device))
            print("Loaded agent from: ", self.resume_path)
        # replay buffer: `save_last_obs` and `stack_num` can be removed together
        # when you have enough RAM
        if self.model == 'CDE':
            buffer = PrioritizedVectorReplayBuffer(
                self.buffer_size,
                buffer_num=len(self.train_envs),
                stack_num=self.history_len,
                alpha=0.6, beta=0.4
            )
        else:
            buffer =  PrioritizedVectorReplayBuffer(
                self.buffer_size,
                buffer_num=len(self.train_envs),
                alpha=0.6, beta=0.4
            )
        # collector
        train_collector = Collector(policy, self.train_envs, buffer, exploration_noise=True)
        test_collector = Collector(policy, self.test_envs, exploration_noise=True)

        # log
        now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        self.algo_name = "dqn"
        log_name = os.path.join(self.task, self.algo_name, str(self.seed), now)
        log_path = os.path.join(self.logdir, log_name)

        # logger
        if self.logger == "wandb":
            logger = WandbLogger(
                save_interval=1,
                name=log_name.replace(os.path.sep, "__"),
                run_id=self.resume_id,
                config=self,
                project=self.wandb_project,
            )
        writer = SummaryWriter(log_path)
        writer.add_text("args", str(self))
        if self.logger == "tensorboard":
            logger = TensorboardLogger(writer)
        else:  # wandb
            logger.load(writer)

        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

        def train_fn(epoch, env_step):
            # nature DQN setting, linear decay in the first 1M steps
            if env_step <= 1e6:
                eps = self.eps_train - env_step / 1e6 * \
                    (self.eps_train - self.eps_train_final)
            else:
                eps = self.eps_train_final
            policy.set_eps(eps)
            if env_step % 1000 == 0:
                logger.write("train/env_step", env_step, {"train/eps": eps})

        def test_fn(epoch, env_step):
            policy.set_eps(self.eps_test)

        def save_checkpoint_fn(epoch, env_step, gradient_step):
            # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
            ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
            torch.save({"model": policy.state_dict()}, ckpt_path)
            return ckpt_path


        # test train_collector and start filling replay buffer
        train_collector.collect(n_step=self.batch_size * self.training_num)
        # trainer
        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            self.epoch,
            self.step_per_epoch,
            self.step_per_collect,
            self.test_num,
            self.batch_size,
            train_fn=train_fn,
            test_fn=test_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            update_per_step=self.update_per_step,
            test_in_train=False,
            resume_from_log=self.resume_id is not None,
            save_checkpoint_fn=save_checkpoint_fn,
        )

        pprint.pprint(result)


if __name__ == "__main__":
    runner = Runner(sys.argv[1])
    runner.test_dqn()
