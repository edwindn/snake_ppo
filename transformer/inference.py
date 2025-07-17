import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

from models import DecisionTransformer

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # must be the same as training config
    config = {
        'lr': 5e-5,
        'batch_size': 16,
        'debug': False,
        'total_trajectories': 16 * 64,
        'max_steps': 500,
        'total_training_runs': 10,
        'save_dir': 'model_checkpoints',
        'logging': True,

        'gamma': 0.9,  # reward discounting
        'action_dim': 2, # for cart pole, need some modifications for continuous action space
        'reward_type': 'always_maximal',  # 'always_maximal' or 'actual'

        'save_training_videos': True,
        'video_folder': 'cartpole_videos',

        'max_reward': 1.0, # placeholder for cartpole max reward per step
        'max_rtg': 10.0, # for cartpole max discounted reward over 500 iters
        'min_state': [-2.4, -5.0, -.2095, -5.0], # cartpole limits -> 5.0 limit is arbitrary for now
        'max_state': [2.4, 5.0, .2095, 5.0],
    }

    gym_environment = 'CartPole-v1'
    env = gym.make(gym_environment, render_mode='rgb_array')
    state_dim = env.observation_space.shape[0]

    model = DecisionTransformer(config['action_dim'], state_dim, device=device)
    model.to(device)
    model.load_model(f"{config['save_dir']}/decision_transformer_ppo.pth")
    model.eval()




