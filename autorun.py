import os
import yaml
import random
import subprocess
import re
import logging
import time
from datetime import datetime
from glob import glob
from copy import deepcopy


def create_config(base_config, **kwargs):
    for (k, v) in kwargs.items():
        base_config[k] = v
    return base_config


def get_free_device(total_cards, just_deployed):
    proc = subprocess.Popen("nvidia-smi", stdout=subprocess.PIPE)
    proc.wait()
    used = [int(i) for i in re.findall(r"(\d)\ *N/A\ *N/A", proc.stdout.read().decode())]
    just_deployed = [int(i) for i in just_deployed if i != '']
    free_cards = list(set(range(total_cards)) - set(used) - set(just_deployed))
    if len(free_cards) != 0:
        return str(free_cards[0])
    else:
        logging.warning("No free GPUs are found, fallback to CPU training")
        return ""


def exec_experiment(conf_fn, device_id):
    cmd = "CUDA_VISIBLE_DEVICES={} python trainer.py {}".format(device_id, conf_fn)
    logging.info(f"Executing command: {cmd}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    return proc


def train_proc_callback(proc):
    proc.wait()
    result = proc.stdout.read().decode()
    result = eval('{' + result.split("{")[1].split("}")[0] + '}')
    logging.info("Done, result is" + str(result))
    return result


def main():
    base_config_path = "config/config-walker.yml"
    actor_type = 'MLP'
    critic_type = 'MLP'
    epoch = 1
    total_cards = 8
    seed_list = [0, 1, 2, 2023]
    deployed_cards = []
    logging.basicConfig(filename=f"autorun-log-{str(datetime.now()).replace(' ', '-')}.log", level=logging.INFO)
    logging.info(f"base_config={base_config_path} actor={actor_type}, critic={critic_type}, seed={seed_list}, epoch={epoch}")
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    running_jobs = []
    
    for seed in seed_list:
        expr_id = random.randint(0, 100000)
        logging.info(f"Experiment ID: {expr_id}")

        conf = create_config(deepcopy(base_config), seed=seed, expr_id=expr_id,
                             actor_type=actor_type, critic_type=critic_type, epoch=epoch)
        conf_fn = "autorunconf/{}.yml".format(expr_id)
        with open(conf_fn, 'w') as f:
            yaml.safe_dump(conf, f)
        logging.info(f"create {conf_fn}, which seed={seed}")

        target_device = get_free_device(total_cards, deployed_cards)
        if target_device != '':
            logging.info(f"get a free GPU device, CUDA:{target_device}")

        proc = exec_experiment(conf_fn, target_device)
        running_jobs.append(proc)
        deployed_cards.append(target_device)
        logging.info(f"GPU {target_device} is already in use\n\n\n")
    

    # RR query until all processes done.
    job_idx = 0
    while len(running_jobs) != 0:
        job = running_jobs[job_idx]
        status = job.poll()
        if status is not None:
            logging.info(f"Subprocess id={job} finished with code {status}!")
            result = train_proc_callback(job)
            base_dir = "log/{}/ppo/{}/".format(conf['task'], seed)
            expr_fn = None
            for fn in os.listdir(base_dir):
                if str(expr_id) in fn:
                    expr_fn = fn
                    break
            expr_log = os.path.join(base_dir, expr_fn)

            with open(expr_log + '/result.yml', 'w') as resultf:
                yaml.safe_dump(result, resultf)
            events_log = glob(expr_log + "/events*")[0]
            logging.info(f"Loss and training curve are stored in {events_log}")

            running_jobs.pop(job_idx)
        # forward to next job
        if len(running_jobs) != 0:
            job_idx = (job_idx + 1) % len(running_jobs)
            time.sleep(1)

if __name__ == '__main__':
    main()
