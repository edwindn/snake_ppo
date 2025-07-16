import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder
from stable_baselines3.common.callbacks import BaseCallback
import time
import os
import logging

# Suppress MoviePy verbose output
logging.getLogger("moviepy").setLevel(logging.ERROR)

class VideoRecorderCallback(BaseCallback):
    def __init__(self, save_freq: int, video_folder: str, video_length: int):
        super(VideoRecorderCallback, self).__init__()
        self.save_freq = save_freq
        self.video_folder = video_folder
        self.video_length = video_length
        self.step_count = 0
        os.makedirs(video_folder, exist_ok=True)

    def _on_step(self) -> bool:
        self.step_count += 1
        if self.step_count % self.save_freq == 0:
            # Create evaluation environment for video recording
            eval_env = make_vec_env("CartPole-v1", n_envs=1)
            prefix = int(time.time())
            video_env = VecVideoRecorder(
                eval_env,
                video_folder=self.video_folder,
                record_video_trigger=lambda x: True,
                video_length=self.video_length,
                name_prefix=f"ppo-cartpole-{prefix}"
            )
            
            # Record video
            obs = video_env.reset()
            for _ in range(self.video_length):
                action, _ = self.model.predict(obs, deterministic=True)
                step_result = video_env.step(action)
                
                # Handle variable step output (Gymnasium vs older Gym)
                if len(step_result) == 5:
                    obs, _, terminated, truncated, _ = step_result
                else:
                    obs, _, done, _ = step_result
                    terminated, truncated = done, False  # Map older 'done' to 'terminated'
                
                if terminated or truncated:
                    obs = video_env.reset()
            
            # Clean up
            video_env.close()
            eval_env.close()
        return True

def train_ppo():
    # Configuration
    config = {
        'save_freq': 10000,  # Record video every 1000 steps
        'total_timesteps': 100000,
        'n_envs': 4,
        'video_length': 200,
        'video_folder': './cartpole_videos/'
    }
    
    # Create training environment
    train_env = make_vec_env(
        "CartPole-v1",
        n_envs=config['n_envs'],
        vec_env_cls=SubprocVecEnv
    )
    
    # Initialize PPO model
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
    
    # Create video recording callback
    video_callback = VideoRecorderCallback(
        save_freq=config['save_freq'],
        video_folder=config['video_folder'],
        video_length=config['video_length']
    )
    
    # Train the model
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=video_callback,
        progress_bar=True
    )
    
    # Save the final model
    model.save("ppo_cartpole")
    train_env.close()
    
    return model

if __name__ == "__main__":
    # Ensure multiprocessing compatibility
    from multiprocessing import freeze_support
    freeze_support()
    
    model = train_ppo()
    
    # Test the trained model
    env = gym.make("CartPole-v1", render_mode="human")
    obs, _ = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()