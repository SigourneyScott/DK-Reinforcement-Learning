from collections import defaultdict
import gymnasium as gym
import numpy as np

class DKAgent:
    def __init__(self, env: gym.Env, 
                 learning_rate: float, 
                 initial_epsilon: float, 
                 epsilon_decay: float,
                 final_epsilon: float, 
                 discount_factor: float = 0.95):
        self.env = env
        self.q_val = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_val[obs]))
        
    def update(self, 
               obs: tuple[int, int, bool], 
               action: int, 
               reward: float, 
               terminated: bool, 
               next_obs: tuple[int, int, bool]):
        future_q_val = (not terminated) * np.max(self.q_val[next_obs])
        temporal_diff = (reward + self.discount_factor * future_q_val - self.q_val[obs][action])
        self.q_val[obs][action] = (self.q_val[obs][action] + self.lr * temporal_diff)
        self.training_error.append(temporal_diff)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

