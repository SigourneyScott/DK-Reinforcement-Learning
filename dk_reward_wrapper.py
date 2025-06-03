import gymnasium as gym
from gymnasium import Wrapper
import ale_py

class DKRewardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_height = 25

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        new_reward = reward

        flattened_obs = obs.flatten().tolist()
        flattened_obs.reverse()
        height = flattened_obs.index(72)//480 #72 is one of the three rgb values of the red of mario's jumpsuit
        new_reward += (self.last_height - height) * 100
        self.last_height = height

        return obs, new_reward, terminated, truncated, info