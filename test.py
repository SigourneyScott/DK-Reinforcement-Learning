import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import FlattenObservation
from collections import defaultdict

from agent import DKAgent  # Ensure DKAgent is defined in dk_agent.py

def get_moving_avgs(arr, window, convolution_mode):
    """
    Compute the moving average over a 1D array.
    """
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window


def main():
    # Create Donkey Kong environment and flatten observations
    env = gym.make("ALE/DonkeyKong-v5", render_mode=None, obs_type="ram")
    env = FlattenObservation(env)
    

    # Initialize agent
    agent = DKAgent(
        env=env,
        learning_rate=0.1,
        initial_epsilon=1.0,
        epsilon_decay=1e-4,
        final_epsilon=0.01,
        discount_factor=0.95
    )

    num_episodes = 1
    episode_rewards = []
    episode_lengths = []

    for ep in range(num_episodes):
        observation, info = env.reset(seed=None)
        # Convert to a hashable state
        state = tuple(observation)
        total_reward = 0.0
        length = 0
        terminated = False
        truncated = False

        # Run one episode
        while not (terminated or truncated):
            action = agent.get_action(state)

            obs, info = env.reset()
            state = tuple(obs)    # now a length‑128 tuple, very cheap
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = tuple(next_obs)

            # Update Q-values and decay epsilon
            agent.update(state, action, reward, terminated, next_state)
            agent.decay_epsilon()

            state = next_state
            total_reward += reward
            length += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(length)

        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1}/{num_episodes}: Reward={total_reward}, Length={length}, Epsilon={agent.epsilon:.3f}")

    # Plot the metrics with moving averages
    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    axs[0].set_title("Episode rewards")
    reward_ma = get_moving_avgs(episode_rewards, rolling_length, "valid")
    axs[0].plot(range(len(reward_ma)), reward_ma)

    axs[1].set_title("Episode lengths")
    length_ma = get_moving_avgs(episode_lengths, rolling_length, "valid")
    axs[1].plot(range(len(length_ma)), length_ma)

    axs[2].set_title("Training Error")
    error_ma = get_moving_avgs(agent.training_error, rolling_length, "same")
    axs[2].plot(range(len(error_ma)), error_ma)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
