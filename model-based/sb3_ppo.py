import matplotlib
matplotlib.use('Agg')  # Use Agg backend for rendering without a display
import matplotlib.pyplot as plt
import imageio
import numpy as np
import time
import os
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import torch
import torch.nn as nn

from nav_env import NavEnv

# Suppress MoviePy verbose output
logging.getLogger("moviepy").setLevel(logging.ERROR)

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class VideoRecorderCallback(BaseCallback):
    def __init__(self, save_freq: int, video_folder: str, video_length: int, max_videos: int = 5):
        super(VideoRecorderCallback, self).__init__()
        self.save_freq = save_freq
        self.video_folder = video_folder
        self.video_length = video_length
        self.max_videos = max_videos
        self.videos_saved = 0
        self.step_count = 0
        os.makedirs(video_folder, exist_ok=True)

    def _on_step(self) -> bool:
        if self.videos_saved >= self.max_videos:
            return True
        self.step_count += 1
        if self.step_count % self.save_freq == 0:
            prefix = int(time.time())
            out_path = os.path.join(self.video_folder, f"rollout_{prefix}.mp4")

            # Create a fresh evaluation env
            eval_env = NavEnv()
            obs, _ = eval_env.reset()
            frames = []

            for i in range(self.video_length):
                action = self.model.predict(obs, deterministic=True)[0]
                obs, *_ = eval_env.step(action)

                state = eval_env.render()

                fig, ax = plt.subplots(figsize=(4, 4))
                ax.scatter(state[0], state[1], c='blue', label='agent')
                ax.scatter(state[2], state[3], c='red', label='goal')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.legend()
                ax.set_title(f"Step {i} dist={obs[4]:.2f}")
                fig.canvas.draw()
                img = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, 1:]
                plt.close(fig)
                frames.append(img)

            eval_env.close()

            imageio.mimwrite(out_path, frames, fps=30)
            print(f"Saved video: {out_path} ({len(frames)} frames)")
            self.videos_saved += 1
        return True


def train_ppo():
    train_timesteps = 100_000

    config = {
        'save_freq': train_timesteps // 5,  # 5 videos over total_timesteps
        'total_timesteps': train_timesteps,
        'video_length': 200,
        'video_folder': './nav_videos/'
    }

    # Create vectorized training env
    train_env = DummyVecEnv([lambda: NavEnv() for _ in range(4)])

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1
    )

    # video_callback = VideoRecorderCallback(
    #     save_freq=config['save_freq'],
    #     video_folder=config['video_folder'],
    #     video_length=config['video_length'],
    #     max_videos=5
    # )

    model.learn(
        total_timesteps=config['total_timesteps'],
        # callback=video_callback,
        progress_bar=True
    )

    model.save("ppo_navigation")
    train_env.close()
    return model


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    os.makedirs('./nav_videos/', exist_ok=True)

    print('Training ...')
    model = train_ppo()
    
    # eval
    save_runs = 4
    print(f'Saving {save_runs} eval runs ...')
    for i in range(save_runs):
        eval_env = NavEnv()
        obs, _ = eval_env.reset()
        frames = []

        for t in range(1000):
            action = model.predict(obs, deterministic=True)[0]
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            state = eval_env.render()

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.scatter(state[0], state[1], c='blue', label='agent')
            ax.scatter(state[2], state[3], c='red', label='goal')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend()
            ax.set_title(f"Step {t} dist={obs[4]:.2f}")
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, 1:]
            plt.close(fig)
            frames.append(img)

            if terminated or truncated:
                print(f"Episode finished after {t+1} steps.")
                break

        eval_env.close()
        out_file = f'./nav_videos/final_rollout_{i+1}.mp4'
        imageio.mimwrite(out_file, frames, fps=30)
        print(f"Saved final video: {out_file} ({len(frames)} frames)")

    print('Done.')
