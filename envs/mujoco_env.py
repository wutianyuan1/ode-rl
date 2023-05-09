import numpy as np
import gym
from gym.spaces import Box


def calculate_dt(obs: np.ndarray):
    MIN_DT = 1
    MAX_DT = 2
    return np.random.randint(MIN_DT, MAX_DT)


class MujocoTimeEnv(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        state_with_time=False
    ):
        super().__init__(env)
        self.time = 0
        self.last_obs = None
        self.state_with_time = state_with_time
        if self.state_with_time:
            self.observation_space = Box(-np.inf, np.inf, (self.observation_space.shape[0] + 1, ))
        self._rewards = []

    def step(self, action):
        dt = calculate_dt(self.last_obs)
        self.env.env.env.env.frame_skip = dt
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.time += dt
        self.last_obs = observation
        if self.state_with_time:
            observation = np.concatenate(([self.time/1000.0], observation))
        self._rewards.append(reward)
        # print("Step with action:", action, "rew=", reward)
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.time = 0
        self.last_obs = obs
        self._rewards = []
        if self.state_with_time:
            obs = np.concatenate(([self.time/1000.0], obs))
        return obs, info
    
    def close(self):
        pass

    def render(self):
        pass


def make_vartime_mujoco_env(task, state_with_time):
    env = gym.make(task)
    return MujocoTimeEnv(env, state_with_time)
