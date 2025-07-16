import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import math
from tqdm import tqdm
import os

from models import DecisionTransformer

# on VM: -- conda activate gpu_env

#TODO:
    # train the transformer in batches for speedup
    
    # allow continous action space

    # normalise states

    # must find maximal reward for specific environment


class Trainer:
    def __init__(self, env, device, model, eval_env=None, **config):
        self.env = env
        self.device = device
        self.model = model
        self.eval_env = eval_env
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-5)
        self.config = config

        if config['save_training_videos']:
            os.makedirs(config['video_folder'], exist_ok=True)
            self.video_folder = config['video_folder']

        self.batch_size = config.get('batch_size')
        self.debug = config.get('debug')
        self.total_trajectories = config.get('total_trajectories')
        self.max_steps = config.get('max_steps')
        self.gamma = config.get('gamma')
        self.action_dim = config.get('action_dim')
        self.reward_type = config.get('reward_type')
        assert self.reward_type in ['always_maximal', 'actual'], "Invalid reward type in config, choose 'always_maximal' or 'actual'."
        if self.reward_type == 'actual':
            raise NotImplementedError # (yet)

        self.max_reward = config['max_reward']
        self.max_rtg = config['max_rtg'] # normalise the rtgs using this value


    def train_iteration(self, rollout_type):
        assert rollout_type in ['random', 'expert'], "Invalid rollout type, choose 'random' or 'expert'."
        losses = []
        episode_lens = []

        self.model.train()

        with tqdm(range(self.total_trajectories)) as pbar:
            for _ in pbar:
                loss, t = self.train_step(rollout_type)
                losses.append(loss)
                episode_lens.append(t)
                avg_loss = sum(losses) / len(losses)
                avg_episode_len = sum(episode_lens) / len(episode_lens)
                pbar.set_description(f"Avg episode len: {avg_episode_len:.2f}, Avg loss: {avg_loss:.4f}")
                # pbar.set_postfix({'loss': loss})
                # pbar.set_postfix({'avg_loss': avg_loss})

    
    def _sinusoidal_timesteps(self, seq_len, dim):
        """Returns a (seq_len, dim) tensor of sinusoidal timestep encodings."""
        position = torch.arange(seq_len, dtype=torch.float32, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=self.device) * (-math.log(10000.0) / dim))
        pe = torch.zeros(seq_len, dim, device=self.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def train_step(self, rollout_type):
        trajectory, t = self.rollout(rollout_type)

        # TODO change to handle batching

        rtgs = torch.tensor([step[0] for step in trajectory], dtype=torch.float32, device=self.device)
        states = torch.tensor(np.array([step[1] for step in trajectory]), dtype=torch.float32, device=self.device)
        actions_long = torch.tensor([step[2] for step in trajectory], dtype=torch.long, device=self.device)

        normalised_rtgs = rtgs / self.max_rtg
        actions_onehot = torch.nn.functional.one_hot(actions_long, num_classes=self.action_dim).float()
        
        # fix batch will remove later
        rtgs = rtgs.unsqueeze(0)
        states = states.unsqueeze(0)
        actions_onehot = actions_onehot.unsqueeze(0)
        actions_long = actions_long.unsqueeze(0)

        seq_len = states.shape[1]
        emb_dim = self.model.embedding_dim
        timesteps = self._sinusoidal_timesteps(seq_len, emb_dim).unsqueeze(0)

        self.optimizer.zero_grad()
        action_preds = self.model(states, actions_onehot, normalised_rtgs, timesteps)
        loss = nn.CrossEntropyLoss()(action_preds.view(-1, action_preds.shape[-1]), actions_long.view(-1))

        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item(), t

    def rollout(self, rollout_type):
        """"
        Returns a list of tensors (or a list of lists of tensors if convert_to_tensors = False)
        of trajectories consisting of S, A, Rtg embedded at each step
        """
        # handle batching here

        return self._collect_trajectory(rollout_type)
    
        # trajectories = []

        # for _ in range(self.total_trajectories):
        #     traj = self._collect_trajectory()
        #     trajectories.append(traj)

        # print(f"Collected {len(trajectories)} trajectories.")
        # return trajectories

    def _collect_trajectory(self, rollout_type):
        trajectory = []
        states = []
        actions = []
        rewards = []

        state, _ = self.env.reset()

        for t in range(self.max_steps):
            states.append(state)

            if rollout_type == 'random':
                # action = np.random.choice(self.action_dim)
                action = self.env.action_space.sample().item()

            elif rollout_type == 'expert':
                if self.reward_type == 'actual':
                    pass
                elif self.reward_type == 'always_maximal':
                    target_rtgs = [1.0] * len(states) # since we normalised the rtgs from 0 to 1

                actions_app = actions + [0] # dummy action for prediction

                state_input = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                rtg_input = torch.tensor(target_rtgs, dtype=torch.float32, device=self.device).unsqueeze(0)
                actions_long = torch.tensor(actions_app, dtype=torch.long, device=self.device).unsqueeze(0)

                actions_onehot = torch.nn.functional.one_hot(actions_long, num_classes=self.action_dim).float()

                timesteps = self._sinusoidal_timesteps(len(states), self.model.embedding_dim).unsqueeze(0)
                if self.debug:
                    print(f"State input shape: {state_input.shape}, Action input shape: {actions_onehot.shape}, Reward input shape: {rtg_input.shape}, Timesteps shape: {timesteps.shape}")
                action_logits = self.model.predict_action(state_input, actions_onehot, rtg_input, timesteps)
                action = torch.argmax(action_logits, dim=-1).item()
            
            next_state, reward, terminated, truncated, _ = self.env.step(action)

            actions.append(action)
            rewards.append(reward)

            done = terminated or truncated
            if done:
                break

            state = next_state

        # compute rewards-to-go
        rtgs = []
        discounted_rew = 0
        for rew in reversed(rewards):
            discounted_rew = rew + discounted_rew * self.gamma
            rtgs.append(discounted_rew)
        rtgs = list(reversed(rtgs))

        for state, action, rtg in zip(states, actions, rtgs):
            trajectory.append([rtg, state, action])

        return trajectory, t

    def test_run(self, run_id=None):
        if self.eval_env is None:
            print("No evaluation environment provided, skipping test run.")
            return

        # Ensure unique video folder per run
        if self.config['save_training_videos']:
            if run_id is not None:
                video_folder = os.path.join(self.video_folder, f"run_{run_id}")
            else:
                video_folder = self.video_folder

            eval_env = RecordVideo(self.eval_env, video_folder=video_folder, episode_trigger=lambda x: True)
            print(f"Recording video to {video_folder}")
        else:
            eval_env = self.eval_env

        states = []
        actions = []
        rewards = []

        state, _ = eval_env.reset()
        for t in range(self.max_steps):
            states.append(state)

            # Model rollout logic (matches _collect_trajectory)
            if self.reward_type == 'actual':
                pass
            elif self.reward_type == 'always_maximal':
                target_rtgs = [1.0] * len(states)  # normalized maximal rtgs

            actions_app = actions + [0]  # dummy action for prediction

            state_input = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            rtg_input = torch.tensor(target_rtgs, dtype=torch.float32, device=self.device).unsqueeze(0)
            actions_long = torch.tensor(actions_app, dtype=torch.long, device=self.device).unsqueeze(0)
            actions_onehot = torch.nn.functional.one_hot(actions_long, num_classes=self.action_dim).float()
            timesteps = self._sinusoidal_timesteps(len(states), self.model.embedding_dim).unsqueeze(0)

            if self.debug:
                print(f"State input shape: {state_input.shape}, Action input shape: {actions_onehot.shape}, Reward input shape: {rtg_input.shape}, Timesteps shape: {timesteps.shape}")

            action_logits = self.model.predict_action(state_input, actions_onehot, rtg_input, timesteps)
            action = torch.argmax(action_logits, dim=-1).item()

            next_state, reward, terminated, truncated, _ = eval_env.step(action)

            actions.append(action)
            rewards.append(reward)

            done = terminated or truncated
            if done:
                break

            state = next_state

        eval_env.close()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = {
        'batch_size': 32,
        'debug': False,
        'total_trajectories': 100,
        'max_steps': 500,
        'total_training_runs': 5,
        'save_dir': 'model_checkpoints',
        'logging': True,

        'gamma': 0.9,  # reward discounting
        'action_dim': 2, # for cart pole, need some modifications for continuous action space
        'reward_type': 'always_maximal',  # 'always_maximal' or 'actual'

        'save_training_videos': True,
        'video_folder': 'cartpole_videos',

        'max_reward': 1.0, # placeholder for cartpole max reward per step
        'max_rtg': 10.0, # for cartpole max discounted reward over 500 iters
        'min_state': [-2.4, -5.0, -.2095, -5.0], # cartpole limits -> 5.0 limit heuristic
        'max_state': [2.4, 5.0, .2095, 5.0],
    }

    os.makedirs(config['save_dir'], exist_ok=True)
    
    gym_environment = 'CartPole-v1'
    env = gym.make(gym_environment)
    state_dim = env.observation_space.shape[0]
    eval_env = gym.make(gym_environment, render_mode="rgb_array")

    model = DecisionTransformer(config['action_dim'], state_dim, device=device)
    model.to(device)

    print("Starting training...")
    trainer = Trainer(env, device, model, eval_env, **config)

    trainer.model.train()
    trainer.train_iteration(rollout_type='random')
    trainer.model.eval()
    trainer.test_run(run_id=0)
    # trainer.model.save_model(f"{config['save_dir']}/decision_transformer_0.pth")

    for run in range(config['total_training_runs'] - 1):
        print(f"--- Training run {run + 1} ---")
        trainer.model.train()
        trainer.train_iteration(rollout_type='expert')
        trainer.model.eval()
        trainer.test_run(run_id=run+1)
        # trainer.model.save_model(f"{config['save_dir']}/decision_transformer_{run + 1}.pth")

    # trainer.model.save_model(f"{config['save_dir']}/decision_transformer_final.pth")
    print("Training complete.")
