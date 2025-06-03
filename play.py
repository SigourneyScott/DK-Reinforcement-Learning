from dk_agent import DonkeyKongAgent


if __name__ == "__main__":
    agent = DonkeyKongAgent(
        num_envs=1, render_mode="human", model_path="ppo_200k_v2"
    )
    agent.load()
    agent.play()
