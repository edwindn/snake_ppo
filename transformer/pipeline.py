import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import math

from rollout import DataRollout
from models import DecisionTransformer


#TODO:
    # train the transformer in batches for speedup


class Trainer:
    def __init__(self, env, device, model, rollout):
        self.env = env
        self.device = device
        self.model = model
        self.rollout = rollout

        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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

        rtgs = torch.tensor([step[0] for step in trajectory], dtype=torch.float32, device=self.device)
        states = torch.tensor([step[1] for step in trajectory], dtype=torch.float32, device=self.device)
        actions_long = torch.tensor([step[2] for step in trajectory], dtype=torch.long, device=self.device)

        # One-hot encode actions for the encoder
        action_dim = self.rollout.action_dim
        actions_onehot = torch.nn.functional.one_hot(actions_long, num_classes=action_dim).float()

        # Add batch dimension
        rtgs = rtgs.unsqueeze(0)
        states = states.unsqueeze(0)
        actions_onehot = actions_onehot.unsqueeze(0)
        actions_long = actions_long.unsqueeze(0)

        seq_len = states.shape[1]
        emb_dim = self.model.embedding_dim
        timesteps = self._sinusoidal_timesteps(seq_len, emb_dim).unsqueeze(0)

        self.optimizer.zero_grad()
        return_preds, state_preds, action_preds = self.model(states, actions_onehot, rtgs, timesteps)
        action_loss = nn.CrossEntropyLoss()(action_preds.view(-1, action_preds.shape[-1]), actions_long.view(-1))

        loss = action_loss

        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('CartPole-v1')
 
    action_dim = 2
    state_dim = env.observation_space.shape[0]

    data_rollout = DataRollout(env, action_dim)
    model = DecisionTransformer(env, action_dim, state_dim, device=device)
    model.to(device)

    print("Starting training...")
    trainer = Trainer(env, device, model, data_rollout)
    trainer.train_iteration(num_steps=1000)
    print("Training complete.")