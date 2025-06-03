PPO_CONFIG = {
    "n_steps": 2048,
    "batch_size": 512,
    "n_epochs": 10,
    "learning_rate": 3e-4,
    "ent_coef": 0.01,
    "clip_range": 0.2,
    "gae_lambda": 0.95,
    "max_grad_norm": 0.5,
    "vf_coef": 0.5,
    "target_kl": 0.015
}

TRAINING_CONFIG = {
    "total_timesteps": 500_000,
    "eval_freq": 10000,
    "n_eval_episodes": 10,
    "save_freq": 50000
}