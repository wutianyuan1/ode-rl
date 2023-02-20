import os
import argparse
import random
import time
import numpy as np
import torch
from mfrl import ModelFreeODERL
import utils

from envs.windygrid_simulator import WindyGridSimulator
from envs.hiv_simulator import HIVSimulator
from envs.acrobot_simulator import AcrobotSimulator
try:
    from envs.half_cheetah_simulator import HalfCheetahSimulator
    from envs.swimmer_simulator import SwimmerSimulator
    from envs.hopper_simulator import HopperSimulator
except:
    print("Couldn't import Mujoco.")

parser = argparse.ArgumentParser('Running model-free RL')
parser.add_argument('--train_env_model', action='store_true', help='train environment model')
parser.add_argument('--num_restarts', type=int, default=0, help='the number of restarts')
parser.add_argument('--trained_model_path', type=str, default='', help='the pre-trained environment model path')
parser.add_argument('--env', type=str, default='acrobot', help='the environment')
parser.add_argument('--seed', type=int, default=2020, help='the random seed')
parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor')
parser.add_argument('--obs_normal', action='store_true', help='whether normalize the observation')
parser.add_argument('--latent_dim', type=int, default=10, help='the latent state dimension')
parser.add_argument('--ode_tol', type=float, default=1e-3, help='the relative error tolerance of ODE networks')
parser.add_argument('--ode_dim', type=int, default=5, help='the number of hidden units in ODE network')
parser.add_argument('--enc_hidden_to_latent_dim', type=int, default=5, help='the number of hidden units for hidden to latent')
parser.add_argument('--lr', type=float, default=9e-4, help='the learning rate for training environment model')
parser.add_argument('--batch_size', type=int, default=32, help='the mini-batch size for training environment model')
parser.add_argument('--epochs', type=int, default=150, help='the number of epochs for training environment model')
parser.add_argument('--iters', type=int, default=12000, help='the number of iterations for training environment model')
parser.add_argument('--eps_decay', type=float, default=1e-4, help='the linear decay rate for scheduled sampling')
parser.add_argument('--max_steps', type=int, help='the max steps for running policy and trajectory generation')
parser.add_argument('--episodes', type=int, default=1000, help='the number of episodes for running policy')
parser.add_argument('--mem_size', type=int, default=int(1e5), help='the size of experience replay buffer')
parser.add_argument('--actor_use_ode', action='store_true', help='whether actor use ode')
parser.add_argument('--critic_use_ode', action='store_true', help='whether critic use ode')
parser.add_argument('--log', action='store_true', help='using logger or print')
args = parser.parse_args()

if not os.path.exists("models/"):
    utils.makedirs("models/")
if not os.path.exists("logs/"):
    utils.makedirs("logs/")
if not os.path.exists("results/"):
    utils.makedirs("results/")

# seed for reproducibility
exp_id = int(random.SystemRandom().random() * 100000)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

if args.env == 'grid':
    simulator = WindyGridSimulator()
elif args.env == 'acrobot':
    simulator = AcrobotSimulator()
elif args.env == 'hiv':
    simulator = HIVSimulator()
elif args.env == 'hiv-pomdp':
    simulator = HIVSimulator(podmp=True)
elif args.env == 'half_cheetah':
    simulator = HalfCheetahSimulator()
elif args.env == 'swimmer':
    simulator = SwimmerSimulator()
elif args.env == 'hopper':
    simulator = HopperSimulator()
else:
    raise NotImplementedError
# simulator.seed(args.seed)

ckpt_path = 'models/{}_{}.ckpt'.format(args.env, exp_id)
if args.log:
    log_path = 'logs/newmodel_log_{}_{}.log'.format(args.env, exp_id)
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
else:
    logger = None
utils.logout(logger, 'Experiment: {}, Environment: {}, Seed: {}'.format(exp_id, repr(simulator),
                                                                                   args.seed))
utils.logout(logger,
             'gamma: {}, latent_dim: {}, lr: {}, batch_size: {}, eps_decay: {}, max steps: {}, '
             'obs_normal: {} actor_use_ode: {}, critic_use_ode: {}'.format(args.gamma, 
             args.latent_dim, args.lr, args.batch_size, args.eps_decay, args.max_steps, args.obs_normal,
             args.actor_use_ode, args.critic_use_ode))
utils.logout(logger, 'CUDA is available: {}'.format(torch.cuda.is_available()))
utils.logout(logger, '*' * 50)

oderl = ModelFreeODERL(simulator,
             gamma=args.gamma,
             mem_size=args.mem_size,
             latent_dim=args.latent_dim,
             batch_size=args.batch_size,
             lr=args.lr,
             ode_tol=args.ode_tol,
             ode_dim=args.ode_dim,
             enc_hidden_to_latent_dim=args.enc_hidden_to_latent_dim,
             eps_decay=args.eps_decay,
             obs_normal=args.obs_normal,
             exp_id=exp_id,
             trained_model_path=args.trained_model_path,
             ckpt_path=ckpt_path,
             logger=logger,
             actor_use_ode=args.actor_use_ode,
             critic_use_ode=args.critic_use_ode,
             seed=args.seed)


dic = {'rewards': [], 'trials': [], 'env_steps': []}

for i in range(args.epochs):
    # MF rollout (only used for model-free policy)
    rewards, eval_reward = oderl.train(int(args.max_steps), args.episodes, cur_epoch=i+1)
    dic['env_steps'].append(args.max_steps * (i + 1))
    dic['rewards'].extend(rewards)
    dic['trials'].append(eval_reward)
    torch.save(dic, 'results/{}_reward_{}.ckpt'.format(args.env, args.num_restarts))
utils.logout(logger, '*' * 10 + ' Done ' + '*' * 10)
