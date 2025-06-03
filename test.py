import gymnasium as gym
import ale_py

env = gym.make("ALE/DonkeyKong-v5", render_mode="human")

# Uncomment the following lines to see the shape of the observation space before and after flattening
# print(env.observation_space.shape)
# wrapped_env = FlattenObservation(env)
# print(wrapped_env.observation_space.shape)

last_height = 25
total_reward = 0

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()  # this is where you would insert your policy
    obs, reward, terminated, truncated, info = env.step(action)

    new_reward = reward

    flattened_obs = obs.flatten().tolist()
    flattened_obs.reverse()
    height = flattened_obs.index(72)//480 #72 is one of the three rgb values of the red of mario's jumpsuit
    new_reward += (last_height - height) * 100
    last_height = height

    total_reward += new_reward
    print(total_reward)

    if terminated or truncated:
        print(total_reward)
        total_reward = 0
        observation, info = env.reset()
env.close()
