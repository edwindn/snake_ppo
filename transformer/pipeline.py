import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import math
from tqdm import tqdm

from models import DecisionTransformer

# on VM: -- conda activate gpu_env

#TODO:
    # train the transformer in batches for speedup
    # allow continous action space

    # normalise the values in the trajectory before embedding

    # must find maximal reward for specific environment


class Trainer:
    def __init__(self, env, device, model, **config):
        self.env = env
        self.device = device
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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

        self.max_reward = 1.0 * self.max_steps # placeholder for cartpole rewards per step
        # self.max_reward = 1.0 # if we normalise


    def train_iteration(self, num_steps, rollout_type):
        assert rollout_type in ['random', 'expert'], "Invalid rollout type, choose 'random' or 'expert'."
        losses = []

        self.model.train()

        with tqdm(range(num_steps)) as pbar:
            for _ in pbar:
                loss = self.train_step(rollout_type)
                losses.append(loss)
                avg_loss = sum(losses) / len(losses)
                pbar.set_postfix({'loss': loss})
                pbar.set_postfix({'avg_loss': avg_loss})

    
    def _sinusoidal_timesteps(self, seq_len, dim):
        """Returns a (seq_len, dim) tensor of sinusoidal timestep encodings."""
        position = torch.arange(seq_len, dtype=torch.float32, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=self.device) * (-math.log(10000.0) / dim))
        pe = torch.zeros(seq_len, dim, device=self.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def train_step(self, rollout_type):
        trajectory = self.rollout(rollout_type)

        # TODO change to handle batching

        rtgs = torch.tensor([step[0] for step in trajectory], dtype=torch.float32, device=self.device)
        states = torch.tensor(np.array([step[1] for step in trajectory]), dtype=torch.float32, device=self.device)
        actions_long = torch.tensor([step[2] for step in trajectory], dtype=torch.long, device=self.device)

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
        action_preds = self.model(states, actions_onehot, rtgs, timesteps)
        loss = nn.CrossEntropyLoss()(action_preds.view(-1, action_preds.shape[-1]), actions_long.view(-1))

        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

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
                action = np.random.choice(self.action_dim)
                # action = self.env.action_space.sample()

            elif rollout_type == 'expert':
                if self.reward_type == 'actual':
                    pass
                elif self.reward_type == 'always_maximal':
                    rewards_app = [self.max_reward] * len(states)

                actions_app = actions + [0] # dummy action for prediction

                state_input = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                reward_input = torch.tensor(rewards_app, dtype=torch.float32, device=self.device).unsqueeze(0) # should be maximal rtgs
                actions_long = torch.tensor(actions_app, dtype=torch.long, device=self.device).unsqueeze(0)
                actions_onehot = torch.nn.functional.one_hot(actions_long, num_classes=self.action_dim).float()

                timesteps = self._sinusoidal_timesteps(len(states), self.model.embedding_dim).unsqueeze(0)
                if self.debug:
                    print(f"State input shape: {state_input.shape}, Action input shape: {actions_onehot.shape}, Reward input shape: {reward_input.shape}, Timesteps shape: {timesteps.shape}")
                action_logits = self.model.predict_action(state_input, actions_onehot, reward_input, timesteps)
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

        return trajectory

    def test_run(self, save_video=True, video_folder: str="cartpole_videos"):
        states = []
        actions = []
        rewards = []

        # Create a new evaluation environment and wrap it for video recording
        eval_env = gym.make(self.env.spec.id, render_mode="rgb_array")
        if save_video:
            eval_env = RecordVideo(eval_env, video_folder=video_folder, episode_trigger=lambda x: True)
            print(f"Recording video to {video_folder}")

        state, _ = eval_env.reset()
        for t in range(self.max_steps):
            states.append(state)

            if self.reward_type == 'actual':
                pass
            elif self.reward_type == 'always_maximal':
                rewards_app = [self.max_reward] * len(states)

            actions_app = actions + [0] # dummy action for prediction

            state_input = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            reward_input = torch.tensor(rewards_app, dtype=torch.float32, device=self.device).unsqueeze(0) # should be maximal rtgs
            actions_long = torch.tensor(actions_app, dtype=torch.long, device=self.device).unsqueeze(0)
            actions_onehot = torch.nn.functional.one_hot(actions_long, num_classes=self.action_dim).float()
            timesteps = self._sinusoidal_timesteps(len(states), self.model.embedding_dim).unsqueeze(0)
            
            action_logits = self.model.predict_action(state_input, actions_onehot, reward_input, timesteps)
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
    is_cuda = torch.cuda.is_available()
    print(f"CUDA available: {is_cuda}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    env = gym.make('CartPole-v1')

    config = {
        'batch_size': 32,
        'debug': False,
        'total_trajectories': 1000,
        'max_steps': 500,
        'gamma': 0.9,  # reward discounting
        'action_dim': 2, # for cart pole, need some modifications for continuous action space
        'reward_type': 'always_maximal',  # 'always_maximal' or 'actual'
    }
 
    state_dim = env.observation_space.shape[0]

    model = DecisionTransformer(config['action_dim'], state_dim, device=device)
    model.to(device)

    print("Starting training...")
    trainer = Trainer(env, device, model, **config)

    trainer.test_run()
    quit()
    trainer.train_iteration(num_steps=1000, rollout_type='expert')
    trainer.model.save_model("decision_transformer_0.pth")

    print("Training complete.")