from dk_agent import DonkeyKongAgent


def create_env_with_wrappers(num_envs, render_mode=None):
    agent = DonkeyKongAgent(num_envs=num_envs, render_mode=render_mode)
    env = agent._make_vector_envs()
    env = DKRewardWrapper(env)
    return env, agent


if __name__ == "__main__":
    env, agent = create_env_with_wrappers(num_envs=4)
    
    agent.train(timesteps=200_000)
    agent.evaluate()
