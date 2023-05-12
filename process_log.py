from glob import glob
from tensorboard.backend.event_processing import event_accumulator
import numpy as np


def read_tensorboard(path):
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    test_rew = [(i.step, i.value) for i in ea.scalars.Items("test/reward")]
    best_rew  = np.max([i[1] for i in test_rew])
    final_rew = test_rew[-1][1]
    return best_rew, final_rew


print("Environment, Actor, Critic, Shared, Random Seed, Start Time, Best Reward, Final Reward, filepath")
for fn in glob("log/*/ppo/*/2305*/events*"):
    best, final = read_tensorboard(fn)
    prefix = fn.split("events.out")[0]
    infolist = fn.split("/")
    task, seed, time = infolist[1], infolist[3], infolist[4]
    with open(prefix + 'config.yml', 'r') as f:
        content = eval(f.read())
    if 'net_type' in content:
        share = False
        actor = 'MLP'
        critic = content['net_type']
    else:
        share = ('individual' not in content)
        critic = content['actor_type']
        critic = content['critic_type']
    print(",".join([str(i) for i in (task, actor, critic, share, seed, time, best, final, fn)]))

