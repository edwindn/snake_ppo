import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import tqdm

from agent import Network, CNNNetwork
from env import Environment

class PPO:
    def __init__(self, network, env, input_dim=None, logging=False):
        """
        Implements PPO-clip algorithm for an arbitrary environment and agent network
        with discrete action space

        Parameters:
            network: the actor / critic network
            env: the environment we train in
            hyperparams: override training hyperparameters
        """
        self.timesteps_per_batch = 200
        self.max_timesteps_per_episode = 50
        self.updates_per_iteration = 5
        self.lr = 5e-4
        self.gamma = 0.99
        self.epsilon = 0.2 # the clip parameter
        self.rnd_prob = 0.5 # prob of taking random actions
        self.min_rnd_prob = 0.05
        self.entropy_loss = 1.0

        self.env = env
        self.obs_dim = env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else None
        self.act_dim = env.action_space.n

        self.logging = logging

        if input_dim is None:
            input_dim = self.obs_dim
        # ! force override
        input_dim = 10
        self.actor = network(input_dim, self.act_dim)
        self.critic = network(input_dim, 1)

        self.log_std = nn.Parameter(torch.zeros(self.act_dim))

        self.actor_optim = torch.optim.Adam(list(self.actor.parameters()) + [self.log_std], self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), self.lr)
    
        self.loss_fn = nn.MSELoss()


    def learn(self, total_timesteps):
        t = 0 # timestep count
        i = 0 # iteration count

        initial_rnd_prob = self.rnd_prob
        save_every = total_timesteps // 10
        next_save = 0

        with tqdm.tqdm(total=total_timesteps, desc="Training Progress", unit="steps") as pbar:
            while t < total_timesteps:
                # --- save a policy rollout at regular intervals ---
                if t >= next_save:
                    eval_env = Environment(**config)
                    eval_env = RecordVideo(
                        eval_env,
                        video_folder='snake_videos',
                        name_prefix=f'training_{t}',
                        episode_trigger=lambda ep_id: True
                    )
                    obs, _ = eval_env.reset()
                    done = False
                    while not done:
                        action, _ = self.get_action(obs, eval_mode=True)
                        obs, _, terminated, truncated, _ = eval_env.step(action)
                        done = terminated or truncated
                    eval_env.close()
                    next_save += save_every
                # --- ---

                obs, acts, logprobs, rtgs, lens, rewards = self.rollout() # batched trajectories

                t += np.sum(lens)
                pbar.update(np.sum(lens))
                i += 1

                self.rnd_prob = max(self.min_rnd_prob, initial_rnd_prob - (initial_rnd_prob - self.min_rnd_prob) * (t/total_timesteps))

                if self.logging:
                    wandb.log({
                        'rnd_prob': self.rnd_prob
                    })

                avg_reward = np.mean([sum(ep_rews) for ep_rews in rewards])
                if self.logging:
                    wandb.log({
                        "avg_reward": avg_reward,
                    })

                V, _ = self.evaluate(obs, acts) # calls value estimator
                A = rtgs - V.detach() # calculate advantage
                A = (A - A.mean()) / (A.std() + 1e-8) # normalize advantage

                for _ in range(self.updates_per_iteration):
                    # update policy by maximising PPO-clip objective
                    # update value function on regression on the total reward

                    V, current_logprobs = self.evaluate(obs, acts)

                    # --- entropy ---
                    obs = obs.view(-1, 10, 10).unsqueeze(1)
                    logits = self.actor(obs)
                    dist = Categorical(logits=logits)
                    entropy = dist.entropy().mean()
                    # ---

                    ratios = torch.exp(current_logprobs - logprobs) # updated policy to current policy

                    loss1 = ratios * A
                    loss2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * A

                    actor_loss = - (torch.min(loss1, loss2)).mean() - self.entropy_loss * entropy
                    critic_loss = self.loss_fn(V, rtgs)

                    if self.logging:
                        wandb.log({
                            'actor_loss': actor_loss.item(),
                            'critic_loss': critic_loss.item()
                        })

                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    self.actor_optim.step()

                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    self.critic_optim.step()
    

    def rollout(self):
        """
        Collect data from running the simulation with updated network
        since PPO is on-policy

        Returns:
            obs
            acts
            logprobs: log probabilities of each action taken; Shape = (num timesteps,)
            rtgs: rewards to go for each timestep of the batch; Shape = (num timesteps,)
            lens: length of the episode for eatch batch; Shape = (num episodes,)
        """
        batch_obs, batch_acts, batch_logprobs, batch_rewards, batch_lens = [], [], [], [], []

        t = 0

        while t < self.timesteps_per_batch:
            episode_rewards = []
            obs, _ = self.env.reset()
            done = False

            for i in range(self.max_timesteps_per_episode):
                t += 1
                batch_obs.append(obs.flatten())
                action, logprob = self.get_action(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                # self.env.render() ######
                done = terminated or truncated

                episode_rewards.append(reward)
                batch_acts.append(action)
                batch_logprobs.append(logprob)

                if self.logging:
                    wandb.log({
                        'score': info['score']
                    })

                if done:
                    break

            batch_lens.append(i+1)
            batch_rewards.append(episode_rewards)

        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float32)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.long) # for discrete action space
        batch_logprobs = torch.tensor(np.array(batch_logprobs), dtype=torch.float32)
        batch_rtgs = self.compute_rtgs(batch_rewards)

        return batch_obs, batch_acts, batch_logprobs, batch_rtgs, batch_lens, batch_rewards
                

    def compute_rtgs(self, rewards):
        """
        Computes RTGs (rewards to go) i.e. total discounted reward from a timestep t to the end of the episode

        Parameters:
            rewards: all rewards in the given batch; Shape = (num episodes, num timesteps per episode)

        Returns:
            rtgs: rewards to go; Shape = (num timesteps in batch,)
        """
        batch_rtgs = []

        for ep_rews in reversed(rewards):
            discounted_rew = 0

            for rew in reversed(ep_rews):
                discounted_rew = rew + discounted_rew * self.gamma
                batch_rtgs.append(discounted_rew)

        # note this assumes the batch loops over the episodes

        batch_rtgs = torch.tensor(batch_rtgs[::-1], dtype=torch.float32)
        return batch_rtgs


    def get_action(self, state, eval_mode=False):
        """
        Queries the actor network, should be called from rollout function

        Returns:
            action: action to take
            logprob: log prob of the action selected from distribution
        """
        # state = torch.tensor(state.flatten(), dtype=torch.float32)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        logits = self.actor(state).flatten()
        dist = Categorical(logits=logits)

        if np.random.rand() > self.rnd_prob or eval_mode:
            action = dist.sample()
        else:
            action = np.random.choice(self.act_dim)
            action = torch.tensor(action)
        logprob = dist.log_prob(action)

        return action.item(), logprob.detach()


    def evaluate(self, states, actions):
        """
        Estimates the value of the state and the log probabilities of each action

        Parameters:
            states: observations (states) from the most recently collected batch (Tensor); Shape: (timesteps, obs dim)
            actions: actions taken from the most recent batch (Tensor); Shape: (timesteps, action dim)

        Returns:
            V: predicted value of the observations
            logprobs: log probs of the actions given the states
        """
        states = states.view(-1, 10, 10).unsqueeze(1)
        V = self.critic(states).squeeze()

        logits = self.actor(states)
        dist = Categorical(logits=logits)
        logprobs = dist.log_prob(actions)

        return V, logprobs
    

    def save_model(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'log_std': self.log_std,
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.log_std = nn.Parameter(checkpoint['log_std'])
    


if __name__ == '__main__':
    config = {
        'input_dim': 10,
        'grid_size': 10,
        'initial_length': 3,
        'debug': False,
        'render_mode': 'rgb_array',
        'logging': False,
        'record_videos': True,
        'total_timesteps': 100_000,
    }

    env = Environment(**config)

    if config['logging']:
        import wandb
        wandb.init(
            project="snake_ppo_testing",
        )

    ppo = PPO(CNNNetwork, env, input_dim=config['grid_size']**2, logging=config['logging'])

    # Remove video wrapper from training env, handled in learn()
    # if config['record_videos']:
    #     def episode_trigger(episode_id):
    #         print(f"Checking episode {episode_id} for video recording")
    #         return episode_id % 100 == 0
    #     env = RecordVideo(env, video_folder='snake_videos', name_prefix='training', episode_trigger=episode_trigger)

    ppo.learn(total_timesteps = config['total_timesteps'])

    # ppo.save_model('checkpoint.pth')

    env.close()
    if config['logging']:
        wandb.finish()

    #--- save a test run ---
    eval_config = config.copy()
    eval_config['render_mode'] = 'rgb_array'
    eval_env = Environment(**eval_config)
    eval_env = RecordVideo(eval_env, video_folder='snake_videos', name_prefix='eval', episode_trigger=lambda ep_id: True)

    for j in range(4):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = ppo.get_action(obs, eval_mode=True)
            obs, _, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated

    eval_env.close()
