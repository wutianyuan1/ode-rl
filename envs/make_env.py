from tianshou.env import ShmemVectorEnv
from envs.norm_state import VectorEnvNormObs
from envs.mujoco_env import make_vartime_mujoco_env
from envs.windygrid import make_windygrid_env
from envs.hivsim import make_hivsim_env
from envs.contwindygrid import make_contwindygrid_env
from functools import partial


def make_env(task, seed, training_num, test_num, obs_norm, state_with_time=False):
    """Wrapper function for Mujoco env.
    If EnvPool is installed, it will automatically switch to EnvPool's Mujoco env.
    :return: a tuple of (single env, training envs, test envs).
    """
    if task == 'WindyGrid':
        env_func = partial(make_windygrid_env, state_with_time)
    elif task == 'HIV':
        env_func = partial(make_hivsim_env, state_with_time)
    elif task == 'ContWindyGrid':
        env_func = partial(make_contwindygrid_env, state_with_time)
    else:
        env_func = partial(make_vartime_mujoco_env, task, state_with_time)

    env = env_func()
    train_envs = ShmemVectorEnv(
        [lambda: env_func() for _ in range(training_num)]
    )
    test_envs = ShmemVectorEnv([lambda: env_func() for _ in range(test_num)])
    train_envs.seed(seed)
    train_envs.get_obs_rms = lambda: None
    test_envs.seed(seed)
    if obs_norm:
        # obs norm wrapper
        train_envs = VectorEnvNormObs(train_envs, state_with_time=state_with_time)
        test_envs = VectorEnvNormObs(test_envs, update_obs_rms=False, state_with_time=state_with_time)
        test_envs.set_obs_rms(train_envs.get_obs_rms())
    return env, train_envs, test_envs
