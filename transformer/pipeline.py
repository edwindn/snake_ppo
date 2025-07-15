import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from datasets import DataLoader
import math

from rollout import DataRollout
from models import DecisionTransformer


class Trainer:
    def __init__(self, env, device, model, rollout):
        self.env = env
        self.device = device
        self.model = model
        self.rollout = rollout

    def train_iteration(self, num_steps):
        losses = []

        self.model.train()

        for _ in range(num_steps):
            loss = self.train_step()
            losses.append(loss)

        avg_loss = sum(losses) / len(losses)
        print(f"Average Loss: {avg_loss:.4f}")

    
    def _sinusoidal_timesteps(self, seq_len, dim):
        """Returns a (seq_len, dim) tensor of sinusoidal timestep encodings."""
        position = torch.arange(seq_len, dtype=torch.float32, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=self.device) * (-math.log(10000.0) / dim))
        pe = torch.zeros(seq_len, dim, device=self.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def train_step(self):
        trajectory = self.rollout.rollout()

        states = torch.tensor([step[0] for step in trajectory], dtype=torch.float32, device=self.device)
        actions = torch.tensor([step[1] for step in trajectory], dtype=torch.long, device=self.device)
        rtgs = torch.tensor([step[2] for step in trajectory], dtype=torch.float32, device=self.device)

        seq_len = states.shape[0]
        emb_dim = states.shape[-1]
        timesteps = self._sinusoidal_timesteps(seq_len, emb_dim, self.device)  # (seq_len, emb_dim)

        self.model.optimizer.zero_grad()
        return_preds, state_preds, action_preds = self.model(states, actions, rtgs, timesteps)
        loss = self.model.loss_fn(torch.cat((return_preds, state_preds, action_preds), dim=1),
                                    torch.cat((rtgs, states, actions), dim=1))

        loss.backward()
        self.model.optimizer.step()

        return loss.detach().cpu().item()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('CartPole-v1')

    data_rollout = DataRollout(env=env)
    model = DecisionTransformer(env, device=device)
    model.to(device)

    trainer = Trainer(env, device, model, data_rollout)
    print("Training complete.")