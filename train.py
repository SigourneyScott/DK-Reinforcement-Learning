from dk_agent import DonkeyKongAgent


if __name__ == "__main__":
    # Initialize and train the agent
    agent = DonkeyKongAgent(num_envs=16, model_path="dk_agent_ppo_1m_wrapped")
    agent.train(timesteps=1_000_000)
    agent.evaluate()
