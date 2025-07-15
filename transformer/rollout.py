import numpy as np
import torch
import gymnasium as gym
from models import CNNEncoder, LinearEncoder
from datasets import DataLoader
import math


# TODO:
    # normalise the values in the trajectory before embedding
    # add actor head
    # add attention mask ?

    # loss only on action predictions


class DataRollout:
    def __init__(self, env, **config):
        self.debug = False

        self.env = env
        self.total_trajectories = config.get('total_trajectories', 1000)
        self.max_steps = 500
        self.gamma = 0.9 # reward discounting
    
    def rollout(self):
        """"
        Returns a list of tensors (or a list of lists of tensors if convert_to_tensors = False)
        of trajectories consisting of S, A, Rtg embedded at each step
        """
        return self._collect_trajectory()
    
        trajectories = []

        for _ in range(self.total_trajectories):
            traj = self._collect_trajectory()
            trajectories.append(traj)

        print(f"Collected {len(trajectories)} trajectories.")
        return trajectories


    def _sinusoidal_positional_encoding(self, seq_len, dim, device):
        """Returns a (seq_len, dim) tensor of sinusoidal positional encodings."""
        position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / dim))
        pe = torch.zeros(seq_len, dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _collect_trajectory(self, convert_to_tensors=False):
        trajectory = []
        state_actions = []
        rewards = []

        state, _ = self.env.reset()

        for t in range(self.max_steps):
            action = np.random.uniform(low=self.action_min, high=self.action_max, size=self.action_dim).tolist()
            next_state, reward, terminated, truncated, _ = self.env.step(action)

            state_actions.append([state, action])
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
            rew_emb = self.rtg_encoder(torch.tensor(discounted_rew))
            rtgs.append(rew_emb)
        rtgs = list(reversed(rtgs))

        # Add sinusoidal positional encodings
        # seq_len = len(state_actions)
        # device = state_actions[0][0].device if hasattr(state_actions[0][0], "device") else "cpu"
        # emb_dim = state_actions[0][0].shape[-1]
        # pos_enc = self._sinusoidal_positional_encoding(seq_len, emb_dim, device)

        # for idx, ((state, action), rtg) in enumerate(zip(state_actions, rtgs)):
        #     pe = pos_enc[idx]
        #     state_pe = state + pe
        #     action_pe = action + pe
        #     rtg_pe = rtg + pe
        #     trajectory.append([rtg_pe, state_pe, action_pe])

        for (state, action), rtg in zip(state_actions, rtgs):
            trajectory.append([rtg, state, action])

        if convert_to_tensors:
            trajectory = torch.stack([torch.stack(tokens, dim=0) for tokens in trajectory], dim=0)
            embedding_dim = trajectory.shape[-1]
            trajectory = trajectory.view(-1, embedding_dim)
            # check this is seq_len * 3, embedding_dim
            if self.debug:
                print(f"Trajectory shape: {trajectory.shape}")

        return trajectory


# Example usage:
if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    data_rollout = DataRollout(env=env)
    trajectories = data_rollout.rollout()