import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DKRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_score = 0
        self.prev_height = 0
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Height-based reward
        height = info.get('y_pos', 0)
        height_reward = (height - self.prev_height) * 0.1
        self.prev_height = height
        
        # Score-based reward
        score = info.get('score', 0)
        score_reward = (score - self.prev_score) * 0.1
        self.prev_score = score
        
        # Survival reward
        survival_reward = 0.1 if not terminated else -1.0
        
        # Combine rewards
        modified_reward = reward + height_reward + score_reward + survival_reward
        
        return obs, modified_reward, terminated, truncated, info