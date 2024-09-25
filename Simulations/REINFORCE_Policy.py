import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import random
import gymnasium as gym
from typing import Tuple
import os

from Simulations import *
from sklearn.preprocessing import normalize
import sys
sys.modules["gym"] = gym

class Policy_Network(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes a neural network that estimates the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        hidden_space1 = 512  # Nothing special with 16, feel free to change
        hidden_space2 = 128
        hidden_space3 = 64 # Nothing special with 32, feel free to change
        self.obs_space_dims = obs_space_dims
        self.action_space_dims = action_space_dims
        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Hardtanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Hardtanh(),
            nn.Linear(hidden_space2, hidden_space3),
            nn.Hardtanh(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space3, action_space_dims)
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space3, action_space_dims)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Conditioned on the observation, returns the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            action_means: predicted mean of the normal distribution
            action_stddevs: predicted standard deviation of the normal distribution
        """
        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )

        return action_means, action_stddevs

class REINFORCE:
   """REINFORCE algorithm."""

   def __init__(self, obs_space_dims: int, action_space_dims: int, episode: int, state_dict: None, optimizer_state_dict: None):
      """Initializes an agent that learns a policy via REINFORCE algorithm [1]
      to solve the task at hand (Inverted Pendulum v4).

      Args:
          obs_space_dims: Dimension of the observation space
          action_space_dims: Dimension of the action space
      """

      # Hyperparameters
      self.learning_rate = 0.001  # Learning rate for policy optimization
      self.gamma = 0.99  # Discount factor
      self.eps = 1e-6  # small number for mathematical stability
      self.episode = episode
      self.obs_space_dims = obs_space_dims
      self.action_space_dims = action_space_dims
      self.state_dict = state_dict
      self.optimizer_state_dict = optimizer_state_dict

      self.probs = []  # Stores probability values of the sampled action
      self.rewards = []  # Stores the corresponding rewards

      self.net = Policy_Network(obs_space_dims, action_space_dims)
      if self.state_dict is not None:
          self.net.load_state_dict(state_dict)
          self.net.eval()

      # self.net = self.net.to(device)
      self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

      if self.optimizer_state_dict is not None:
          self.optimizer.load_state_dict(self.optimizer_state_dict)

   def sample_action(self, state: np.ndarray) -> float:
      """Returns an action, conditioned on the policy and observation.

      Args:
          state: Observation from the environment

      Returns:
          action: Action to be performed
      # """

      state = torch.tensor(np.array([state]))

      action_means, action_stddevs = self.net(state)

      # create a normal distribution from the predicted
      #   mean and standard deviation and sample an action
      distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
      action = distrib.sample()
      prob = distrib.log_prob(action)

      action = action.numpy()

      self.probs.append(prob)

      return action

   def update(self):
      """Updates the policy network's weights."""
      running_g = 0
      gs = []

      # Discounted return (backwards) - [::-1] will return an array in reverse
      for R in self.rewards[::-1]:
         running_g = R + self.gamma * running_g
         gs.insert(0, running_g)

      deltas = torch.tensor(gs)

      loss = 0
      # minimize -1 * prob * reward obtained
      for log_prob, delta in zip(self.probs, deltas):
         loss += log_prob.mean() * delta * (-1)

      # Update the policy network
      self.optimizer.zero_grad()
      loss.backward()
      # Makes learning for stable
      torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.01)
      self.optimizer.step()

      # torch.save({
      #     'epoch': self.episode,
      #     'model_state_dict': self.net.state_dict(),
      #     'optimizer_state_dict': self.optimizer.state_dict(),
      #     'loss': loss
      # }, 'model.pt')


      # Empty / zero out all episode-centric/related variables
      self.probs = []
      self.rewards = []


def train(environment_name, save_path, episode_steps=350):
    env = gym.make(environment_name, render_mode="human", max_episode_steps=episode_steps)

    total_num_episodes = int(3000000)  # Total number of episodes
    # Observation-space of InvertedPendulum-v4 (4)
    obs_space_dims = env.observation_space.shape[0]
    # Action-space of InvertedPendulum-v4 (1)
    action_space_dims = env.action_space.shape[0]

    options = {
        "initial_state_dict":
            {
                "qpos": env.get_env_state()['qpos'],
                "qvel": env.get_env_state()['qvel'],
                "door_body_pos": env.get_env_state()['door_body_pos']
            }

    }
    best_reward_path = os.path.join(save_path, "Checkpoints")
    best_reward = 0.0

    for seed in [1]:  # Fibonacci seeds

        episode_score_df = pd.DataFrame(columns = ['Episode', 'Score'])
        true_episode = 0

        # set seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Reinitialize agent every seed
        agent = REINFORCE(obs_space_dims, action_space_dims, 1, None, None)
        score = 0
        # track scores
        scores = []
        curr_step = 0
        MAX_STEPS = episode_steps

        state, info = env.reset(seed=seed, options=options)

        for episode in range(1, total_num_episodes):
            curr_step += 1
            # gymnasium v26 requires users to set seed while resetting the environment
            action = agent.sample_action(normalize([state]).squeeze())

            # execute action
            state, reward, terminated, truncated, info = env.step(action)
            done = True if terminated or truncated else False

            agent.rewards.append(reward)

            # track rewards
            score += reward

            if float(reward) >= best_reward or float(reward) == 10.0:
                best_reward = reward
                filename = "open_door_REINFORCE_checkpoint" + "_seed_" + str(seed)

                environment_state = {'epoch': true_episode, 'state_dict': agent.net.state_dict(),
                         'optimizer': agent.optimizer.state_dict(), 'action': action}
                checkpoint_info = {'seed': seed, 'epoch': true_episode, 'reward': reward}

                torch.save(environment_state, best_reward_path + "/" + filename + ".pth")

                with open(best_reward_path + "/" + filename + ".json", 'w') as json_file:
                    json.dump(checkpoint_info, json_file)

                print('New better accuracy: ', reward)

            # end episode
            if done or curr_step >= MAX_STEPS:
                print(f'Episode: {true_episode} Reward: {reward}')

                episode_score_df = episode_score_df.append({'Episode' : true_episode, 'Score' : score}, ignore_index=True)
                episode_score_df.to_csv('REINFORCE_episode_score.csv', index=False)
                state, _ = env.reset(options=options, seed=seed)
                curr_step = 0
                scores.append(score)
                score = 0
                # move into new state
                agent.update()
                true_episode += 1
                continue





def continue_train(environment_name, checkpoint_path, episode_steps=350):
    env = gym.make(environment_name, render_mode="human", max_episode_steps=episode_steps)


    total_num_episodes = int(3000000)  # Total number of episodes
    # Observation-space of InvertedPendulum-v4 (4)
    obs_space_dims = env.observation_space.shape[0]
    # Action-space of InvertedPendulum-v4 (1)
    action_space_dims = env.action_space.shape[0]

    options = {
                "initial_state_dict" :
                {
                    "qpos": env.get_env_state()['qpos'],
                    "qvel": env.get_env_state()['qvel'],
                    "door_body_pos": env.get_env_state()['door_body_pos']
                }

             }
    best_reward_path = "/Simulations/OpenDoor/Checkpoints"

    best_reward = 10.0

    for seed in [1]:  # Fibonacci seeds
        episode_score_df = pd.read_csv('/Simulations/OpenDoor/REINFORCE_episode_score.csv')
        true_episode = len(episode_score_df['Score'])

        # set seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # solved environment score
        SOLVED_SCORE = 10.0

        load_checkpoint = torch.load(checkpoint_path)
        state_dict = load_checkpoint['state_dict']
        optimizer_state_dict = load_checkpoint['optimizer']

        # Reinitialize agent every seed
        agent = REINFORCE(obs_space_dims, action_space_dims, 1, state_dict=state_dict, optimizer_state_dict=optimizer_state_dict)
        score = 0
        # track scores
        scores = []
        curr_step = 0
        MAX_STEPS = episode_steps

        state, info = env.reset(seed=seed, options=options)

        for episode in range(1, total_num_episodes):
            curr_step += 1
            # gymnasium v26 requires users to set seed while resetting the environment
            action = agent.sample_action(normalize([state]).squeeze())

            # execute action
            state, reward, terminated, truncated, info = env.step(action)
            done = True if terminated or truncated else False

            agent.rewards.append(reward)

            # track rewards
            score += reward

            if float(reward) >= best_reward:
                filename = "open_door_REINFORCE_checkpoint_round_2" + "_seed_" + str(seed)

                environment_state = {'epoch': episode, 'state_dict': agent.net.state_dict(),
                         'optimizer': agent.optimizer.state_dict(), 'action': action}
                checkpoint_info = {'seed': seed, 'epoch': episode, 'reward': reward}

                torch.save(environment_state, best_reward_path + "/" + filename + ".pth")

                with open(best_reward_path + "/" + filename + ".json", 'w') as json_file:
                    json.dump(checkpoint_info, json_file)

                print('New better accuracy: ', reward)

            # end episode
            if done or curr_step >= MAX_STEPS:
                print(f'Episode: {true_episode} Reward: {reward}')

                episode_score_df = episode_score_df.append({'Episode' : true_episode, 'Score' : score}, ignore_index = True)
                episode_score_df.to_csv('REINFORCE_episode_score.csv', index=False)
                state, _ = env.reset(options=options, seed=seed)
                curr_step = 0
                scores.append(score)
                score = 0
                # move into new state
                agent.update()
                true_episode += 1
                continue





def test(environment_name, checkpoint_path, save_path, episode_steps=200, total_num_episodes=5):
    env = gym.make(environment_name, render_mode="human", max_episode_steps=episode_steps)

    seed = 1
    # Observation-space of InvertedPendulum-v4 (4)
    obs_space_dims = env.observation_space.shape[0]
    # Action-space of InvertedPendulum-v4 (1)
    action_space_dims = env.action_space.shape[0]

    load_checkpoint = torch.load(checkpoint_path)
    state_dict = load_checkpoint['state_dict']
    optimizer_state_dict = load_checkpoint['optimizer']

    options = {
        "initial_state_dict":
            {
                "qpos": env.get_env_state()['qpos'],
                "qvel": env.get_env_state()['qvel'],
                "door_body_pos": env.get_env_state()['door_body_pos']
            }

    }

    # Reinitialize PPO_policy every seed
    agent = REINFORCE(obs_space_dims, action_space_dims, 1, state_dict=state_dict,
                      optimizer_state_dict=optimizer_state_dict)

    state, _ = env.reset(seed=seed, options=options)
    episode = 0
    score = 0
    curr_step = 0
    scores = []
    highest_rewards = []
    num_episodes = int(3000000)

    for i in range(1, num_episodes):
        highest_reward = 0
        if episode == total_num_episodes:
            break

        curr_step += 1
        action = agent.sample_action(normalize([state]).squeeze())
        new_state, reward, terminated, truncated, _ = env.step(action)
        done = True if terminated or truncated or reward == 10.0 else False
        if reward >= highest_reward:
            highest_reward = reward

        score += reward

        if reward == 10.0:
            print(reward)

        if done or curr_step >= episode_steps:
            print('Episode: ', episode)
            episode += 1
            scores.append(score)
            highest_rewards.append(highest_reward)
            new_state, _ = env.reset(seed=seed, options=options)
            curr_step = 0
            score = 0

        state = new_state

    df = pd.DataFrame(scores, columns =['Scores'])
    df.to_csv(os.path.join(save_path, 'REINFORCE_test.csv'), index=False)
    df = pd.DataFrame(highest_rewards, columns=['Highest Reward'])
    df.to_csv(os.path.join(save_path, 'REINFORCE_test_highest_rewards.csv'), index=False)
    sys.exit()
