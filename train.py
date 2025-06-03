from dk_agent import DonkeyKongAgent


if __name__ == "__main__":
    agent = DonkeyKongAgent(num_envs=4, model_path="dk_agent_ppo_1m_reward_wrapper")
    agent.train(timesteps=1_000_000)
    agent.evaluate()
