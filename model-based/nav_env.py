# import gym
import gymnasium as gym
from gymnasium import spaces

import numpy as np
import math

class NavEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self.ndim = 2
        self.max_size = 250
        self.max_step_size = 10
        self.max_distance = math.sqrt(self.max_size**2 + self.max_size**2)
        self.max_steps = 200 # encourage short episodes, 15 steps should be enough
        self.current_step = 0
        self.hit_rate = 0

        self.target_tolerance = 10

        self.action_space = spaces.Box(
            low = -self.max_step_size,
            high = self.max_step_size,
            shape=(self.ndim,),
            dtype=np.float32
        )

        # replace with image later on
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(2*self.ndim + 1,),
            dtype=np.float32
        )

        self.reset()

    def _distance(self):
        return math.sqrt(pow((self.target_x - self.agent_x), 2) + pow(self.target_y - self.agent_y, 2))

    def _get_observation(self):
        obs = [self.agent_x, self.agent_y, self.target_x, self.target_y, self._distance()]
        return obs

    def _normalise_observation(self, obs):
        obs = [o/self.max_size for o in obs[:-1]] + [obs[-1]/self.max_distance]
        return np.array(obs, dtype=np.float32)

    def random_action(self):
        return self.action_space.sample()

    def render(self):
        obs = self._get_observation()
        obs = self._normalise_observation(obs)
        return obs[:-1]

    def reset(self, seed=None):
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.current_step = 0

        self.agent_x = np.random.uniform(0, self.max_size)
        self.agent_y = np.random.uniform(0, self.max_size)
        self.target_x = np.random.uniform(0, self.max_size)
        self.target_y = np.random.uniform(0, self.max_size)

        if self._distance() < self.target_tolerance * 1.1:
            self.target_x = np.random.uniform(0, self.max_size)
            self.target_y = np.random.uniform(0, self.max_size)

        obs = self._get_observation()
        return self._normalise_observation(obs), None

    def step(self, action):
        self.current_step += 1

        terminated = False
        truncated = False

        step_x, step_y = action
        self.agent_x += step_x
        self.agent_y += step_y
        self.agent_x = np.clip(self.agent_x, 0, self.max_size)
        self.agent_y = np.clip(self.agent_y, 0, self.max_size)

        if self._distance() < self.target_tolerance:
            reward = 10.0
            terminated = True
            print(f"Reached target in {self.current_step} steps")
        elif self.current_step >= self.max_steps:
            reward = -1.0
            truncated = True
            # print(f"Episode truncated after {self.current_step} steps")
        else:
            reward = 1.0 - (self._distance() / self.max_distance)
            # print(f"Reward for this step: {reward}")

        obs = self._get_observation()
        obs = self._normalise_observation(obs)

        return obs, reward, terminated, truncated, {"info": None}
