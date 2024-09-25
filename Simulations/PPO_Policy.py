from __future__ import annotations
import torch
import torch.nn as nn
import random
import sys
import gymnasium as gym
import json
import os
import time
import torch.nn.functional as F
from torch.distributions import Normal
from Simulations import *
import numpy as np
import pandas as pd
from collections import deque
from sklearn.preprocessing import normalize

# https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-2-4-f9d8b8aa938a
# https://arxiv.org/pdf/1707.06347
# https://github.com/chengxi600/RLStuff/blob/master/Policy%20Optimization%20Algorithms/PPO_Continuous.ipynb

# if GPU is to be used
# device = torch.device("mps")

# class for actor-critic network
class ActorCriticNetwork(nn.Module):

    def __init__(self, obs_space, action_space):
        '''
        Args:
        - obs_space (int): observation space
        - action_space (int): action space

        '''
        super(ActorCriticNetwork, self).__init__()
        self.obs_space = obs_space
        self.action_space = action_space

        self.actor = nn.Sequential(
            nn.Linear(obs_space, 512),
            nn.Hardtanh(),
            nn.Linear(512, 128),
            nn.Hardtanh(),
            nn.Linear(128, 64),
            nn.Hardtanh(),
            nn.Linear(64, action_space))

        self.critic = nn.Sequential(
            nn.Linear(obs_space, 512),
            nn.Hardtanh(),
            nn.Linear(512, 128),
            nn.Hardtanh(),
            nn.Linear(128, 64),
            nn.Hardtanh(),
            nn.Linear(64, 1))

    def forward(self):
        ''' Not implemented since we call the individual actor and critc networks for forward pass
        '''
        raise NotImplementedError

    def select_action(self, state):
        ''' Selects an action given current state
        Args:
        - network (Torch NN): network to process state
        - state (Array): Array of action space in an environment

        Return:
        - (int): action that is selected
        - (float): log probability of selecting that action given state and network
        '''

        # convert state to float tensor, add 1 dimension, allocate tensor on device
        state = torch.from_numpy(state).float().unsqueeze(0)

        # use network to predict action probabilities
        actions = self.actor(state)

        # sample an action using the Gaussian distribution
        m = Normal(actions, 0.6)
        actions = m.sample()

        # return action
        return actions.detach().numpy().squeeze(0), m.log_prob(actions)

    def evaluate_action(self, states, actions):
        ''' Get log probability and entropy of an action taken in given state
        Args:
        - states (Array): array of states to be evaluated
        - actions (Array): array of actions to be evaluated

        '''

        # convert state to float tensor, add 1 dimension, allocate tensor on device
        states_tensor = torch.stack([torch.from_numpy(state).float().unsqueeze(0) for state in states]).squeeze(1)

        # use network to predict action probabilities
        actions_probs = self.actor(states_tensor)

        # get probability distribution
        m = Normal(actions_probs, 0.1)

        # return log_prob and entropy
        return m.log_prob(torch.Tensor(actions)), m.entropy()


# Proximal Policy Optimization
class PPO_policy():

    def __init__(self, γ, ϵ, β, δ, c1, c2, k_epoch, obs_space, action_space, α_θ, αv, state_dict_actor=None,
                 state_dict_critic=None, optimizer_state_dict=None, eval_mode=False):
        '''
        Args:
        - γ (float): discount factor
        - ϵ (float): soft surrogate objective constraint
        - β (float): KL (Kullback–Leibler) penalty
        - δ (float): KL divergence adaptive target
        - c1 (float): value loss weight
        - c2 (float): entropy weight
        - k_epoch (int): number of epochs to optimize
        - obs_space (int): observation space
        - action_space (int): action space
        - α_θ (float): actor learning rate
        - αv (float): critic learning rate

        '''
        self.γ = γ
        self.ϵ = ϵ
        self.β = β
        self.δ = δ
        self.c1 = c1
        self.c2 = c2
        self.k_epoch = k_epoch
        self.state_dict_actor = state_dict_actor
        self.state_dict_critic = state_dict_critic
        self.optimizer_state_dict = optimizer_state_dict
        self.actor_critic = ActorCriticNetwork(obs_space, action_space)
        self.eval_mode = eval_mode
        if self.state_dict_actor is not None and self.state_dict_critic is not None:
            self.actor_critic.actor.load_state_dict(state_dict_actor)
            self.actor_critic.critic.load_state_dict(state_dict_critic)

        self.optimizer = torch.optim.Adam([
            {'params': self.actor_critic.actor.parameters(), 'lr': α_θ},
            {'params': self.actor_critic.critic.parameters(), 'lr': αv}
        ])
        if self.optimizer_state_dict is not None:
            self.optimizer.load_state_dict(self.optimizer_state_dict)

        if self.eval_mode:
            self.actor_critic.actor.eval()
            self.actor_critic.critic.eval()

        # buffer to store current batch
        self.batch = []

    def process_rewards(self, rewards, terminals):
        ''' Converts our rewards history into cumulative discounted rewards
        Args:
        - rewards (Array): array of rewards

        Returns:
        - G (Array): array of cumulative discounted rewards
        '''
        # Calculate Gt (cumulative discounted rewards)
        G = []

        # track cumulative reward
        total_r = 0

        # iterate rewards from Gt to G0
        for r, done in zip(reversed(rewards), reversed(terminals)):

            # Base case: G(T) = r(T)
            # Recursive: G(t) = r(t) + G(t+1)^DISCOUNT
            total_r = r + total_r * self.γ

            # no future rewards if current step is terminal
            if done:
                total_r = r

            # add to front of G
            G.insert(0, total_r)

        # whitening rewards
        G = torch.tensor(G)
        G = (G - G.mean()) / G.std()

        return G

    def kl_divergence(self, old_lps, new_lps):
        ''' Calculate distance between two distributions with KL divergence
        Args:
        - old_lps (Array): array of old policy log probabilities
        - new_lps (Array): array of new policy log probabilities
        '''

        # track kl divergence
        total = 0

        # sum up divergence for all actions
        for old_lp, new_lp in zip(old_lps, new_lps):
            # same as old_lp * log(old_prob/new_prob) cuz of log rules
            total += old_lp * (old_lp - new_lp)

        return total

    def penalty_update(self):
        ''' Update policy using surrogate objective with adaptive KL penalty
        '''

        # get items from current batch
        states = [sample[0] for sample in self.batch]
        actions = [sample[1] for sample in self.batch]
        rewards = [sample[2] for sample in self.batch]
        old_lps = [sample[3] for sample in self.batch]
        terminals = [sample[4] for sample in self.batch]

        # calculate cumulative discounted rewards
        Gt = self.process_rewards(rewards, terminals)

        # track divergence
        divergence = 0

        # perform k-epoch update
        for epoch in range(self.k_epoch):
            # get ratio
            new_lps, entropies = self.actor_critic.evaluate_action(states, actions)
            # same as new_prob / old_prob
            ratios = torch.exp(new_lps - torch.Tensor(old_lps))

            # compute advantages
            states_tensor = torch.stack([torch.from_numpy(state).float().unsqueeze(0) for state in states]).squeeze(1)
            vals = self.actor_critic.critic(states_tensor).squeeze(1).detach()
            advantages = Gt - vals

            # get loss with adaptive kl penalty
            divergence = self.kl_divergence(old_lps, new_lps).detach()
            loss = -ratios * advantages + self.β * divergence

            # SGD via Adam
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # update adaptive penalty
        if divergence >= 1.5 * self.δ:
            self.β *= 2
        elif divergence <= self.δ / 1.5:
            self.β /= 2

        # clear batch buffer
        self.batch = []

    def clipped_update(self):
        ''' Update policy using clipped surrogate objective
        '''
        # get items from trajectory
        states = [sample[0] for sample in self.batch]
        actions = [sample[1] for sample in self.batch]
        rewards = [sample[2] for sample in self.batch]
        old_lps = [sample[3] for sample in self.batch]
        terminals = [sample[4] for sample in self.batch]

        # calculate cumulative discounted rewards
        Gt = self.process_rewards(rewards, terminals)

        # perform k-epoch update
        for epoch in range(self.k_epoch):
            # get ratio
            new_lps, entropies = self.actor_critic.evaluate_action(states, actions)

            ratios = torch.exp(new_lps - torch.stack(old_lps).squeeze(1).detach())
            # self.actor_critic.actor(torch.from_numpy(states[0]).float())
            # compute advantages
            states_tensor = torch.stack([torch.from_numpy(state).float().unsqueeze(0) for state in states]).squeeze(1)
            vals = self.actor_critic.critic(states_tensor).squeeze(1).detach()
            advantages = Gt - vals

            # clip surrogate objective
            surrogate1 = torch.clamp(ratios, min=1 - self.ϵ, max=1 + self.ϵ) * advantages.unsqueeze(0).T
            surrogate2 = ratios * advantages.unsqueeze(0).T

            # loss, flip signs since this is gradient descent
            loss = -torch.min(surrogate1, surrogate2) + self.c1 * F.mse_loss(Gt, vals) - self.c2 * entropies

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # clear batch buffer
        self.batch = []


# Open Door train
def train(environment_name, path, episode_steps=50, stop_episodes=None, csv_filename='PPO_episode_score.csv', checkPoint_name="PPO_checkpoint"):
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
    best_reward_path = os.path.join(path, "Checkpoints")
    os.makedirs(best_reward_path, exist_ok=True)

    best_reward = 0
    for seed in [1]:  # Fibonacci seeds

        episode_score_df = pd.DataFrame(columns=['Episode', 'Score'])
        true_episode = 0

        # set seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Reinitialize PPO_policy every seed
        ppo_policy = PPO_policy(γ=0.99, ϵ=0.2, β=1, δ=0.01, c1=0.5, c2=0.01, k_epoch=40,
                                obs_space=obs_space_dims, action_space=action_space_dims, α_θ=0.001, αv=0.001)

        # track scores
        scores = []

        # recent 200 scores
        recent_scores = deque(maxlen=200)

        # reset environment, initiate variables
        state, _ = env.reset(options=options, seed=seed)
        curr_step = 0
        MAX_STEPS = episode_steps
        # batch training size
        BATCH_SIZE = 1600

        for episode in range(1, total_num_episodes):
            if stop_episodes is not None:
                if true_episode > stop_episodes:
                    sys.exit()

            episode_rewards = []
            curr_step += 1
            # select action
            action, lp = ppo_policy.actor_critic.select_action(normalize([state]).squeeze())

            # execute action
            state, reward, terminated, truncated, _ = env.step(action)
            term_trun = True if terminated or truncated else False
            done = True if term_trun or reward==10.0 else False

            # track rewards
            scores.append(reward)

            # store into trajectory
            ppo_policy.batch.append([state, action, reward, lp, done])

            # optimize surrogate
            if episode % BATCH_SIZE == 0:
                ppo_policy.clipped_update()

            if float(reward) >= best_reward:
                best_reward = reward
                filename = checkPoint_name + "_seed_" + str(seed)

                episode_state = {'episode': true_episode, 'state_dict_actor': ppo_policy.actor_critic.actor.state_dict(),
                'state_dict_critic': ppo_policy.actor_critic.critic.state_dict(),
                         'optimizer': ppo_policy.optimizer.state_dict(), 'action': action}
                checkpoint_info = {'seed': seed, 'episode': true_episode, 'reward': reward}

                torch.save(episode_state, best_reward_path + "/" + filename + ".pth")

                with open(best_reward_path + "/" + filename + ".json", 'w') as json_file:
                    json.dump(checkpoint_info, json_file)

                print('New better accuracy: ', reward)



            # end episode
            if done or curr_step >= MAX_STEPS:
                print(f'Episode: {true_episode} Reward: {reward}')
                row = pd.DataFrame({'Episode': [true_episode], 'Score': [sum(scores)]})
                episode_score_df = pd.concat([episode_score_df, row], ignore_index=True)
                episode_score_df.to_csv(os.path.join(path, csv_filename), index=False)
                state, _ = env.reset(options=options, seed=seed)
                curr_step = 0
                scores = []
                true_episode += 1

                continue


def continue_train(environment_name, checkpoint_path, save_path, current_best_reward=0, episode_steps=200, round=2):
    env = gym.make(environment_name, render_mode="human", max_episode_steps=episode_steps)


    total_num_episodes = int(300000000)  # Total number of episodes
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
    best_reward_path = os.path.join(save_path, "Checkpoints")

    best_reward = current_best_reward
    for seed in [1]:  # Fibonacci seeds
        episode_score_df = pd.read_csv(os.path.join(save_path, 'PPO_episode_score.csv'))
        true_episode = len(episode_score_df)

        # set seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        load_checkpoint = torch.load(checkpoint_path)
        state_dict_actor = load_checkpoint['state_dict_actor']
        state_dict_critic = load_checkpoint['state_dict_critic']
        optimizer_state_dict = load_checkpoint['optimizer']

        # Reinitialize PPO_policy every seed
        ppo_policy = PPO_policy(γ=0.99, ϵ=0.2, β=1, δ=0.01, c1=0.5, c2=0.01, k_epoch=40,
                                obs_space=obs_space_dims, action_space=action_space_dims, α_θ=0.001, αv=0.001, state_dict_actor=state_dict_actor, state_dict_critic=state_dict_critic, optimizer_state_dict=optimizer_state_dict)

        # track scores
        scores = []

        # recent 100 scores
        recent_scores = deque(maxlen=200)

        # reset environment, initiable variables
        state, _ = env.reset(options=options, seed=seed)
        score = 0
        curr_step = 0
        MAX_STEPS = episode_steps
        # batch training size
        BATCH_SIZE = 1600

        for episode in range(1, total_num_episodes):
            episode_rewards = []
            curr_step += 1
            # select action
            action, lp = ppo_policy.actor_critic.select_action(normalize([state]).squeeze())

            # execute action
            state, reward, terminated, truncated, _ = env.step(action)
            done = True if terminated or truncated else False

            # track rewards
            score += reward

            # store into trajectory
            ppo_policy.batch.append([state, action, reward, lp, done])

            # optimize surrogate
            if episode % BATCH_SIZE == 0:
                ppo_policy.clipped_update()

            if float(reward) >= best_reward or float(reward) == 10.0:
                best_reward = reward
                filename = f"round_{round}_PPO_checkpoint" + "_seed_" + str(seed)

                episode_state = {'episode': true_episode, 'state_dict_actor': ppo_policy.actor_critic.actor.state_dict(),
                'state_dict_critic': ppo_policy.actor_critic.critic.state_dict(),
                         'optimizer': ppo_policy.optimizer.state_dict(), 'action': action}
                checkpoint_info = {'seed': seed, 'episode': true_episode, 'reward': reward}

                torch.save(episode_state, best_reward_path + "/" + filename + ".pth")

                with open(best_reward_path + "/" + filename + ".json", 'w') as json_file:
                    json.dump(checkpoint_info, json_file)

                print('New better accuracy: ', reward)

            # end episode
            if done or curr_step >= MAX_STEPS:
                print(f'Episode: {true_episode} Reward: {reward}')

                row = pd.DataFrame({'Episode': [true_episode], 'Score': [score]})
                episode_score_df = pd.concat([episode_score_df, row], ignore_index=True)
                episode_score_df.to_csv(os.path.join(save_path, 'PPO_episode_score.csv'), index=False)

                state, _ = env.reset(options=options, seed=seed)
                curr_step = 0
                scores.append(score)
                recent_scores.append(score)
                score = 0
                true_episode += 1

                continue


def test(environment_name, checkpoint_path, save_path, episode_steps=50, seed=1, total_num_episodes=5, csv_episode_name='PPO_test_episode.csv', csv_highest_reward_name='PPO_test_highest_reward.csv'):
    env = gym.make(environment_name, render_mode="human", max_episode_steps=episode_steps)

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
    # set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    load_checkpoint = torch.load(checkpoint_path)
    state_dict_actor = load_checkpoint['state_dict_actor']
    state_dict_critic = load_checkpoint['state_dict_critic']
    optimizer_state_dict = load_checkpoint['optimizer']

    # Reinitialize PPO_policy every seed
    ppo_policy = PPO_policy(γ=0.99, ϵ=0.2, β=1, δ=0.01, c1=0.5, c2=0.01, k_epoch=40,
                            obs_space=obs_space_dims, action_space=action_space_dims, α_θ=0.001, αv=0.001,
                            state_dict_actor=state_dict_actor, state_dict_critic=state_dict_critic,
                            optimizer_state_dict=optimizer_state_dict, eval_mode=True)

    scores = []
    highest_rewards = []
    state, _ = env.reset(seed=seed, options=options)
    episode = 0
    score = 0
    curr_step = 0
    num_episodes = int(3000000)

    for i in range(1, num_episodes):
        highest_reward = 0
        if episode == total_num_episodes:
            break

        curr_step += 1
        action, lp = ppo_policy.actor_critic.select_action(normalize([state]).squeeze())
        state, reward, terminated, truncated, _ = env.step(action)
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
            state, _ = env.reset(options=options, seed=seed)
            curr_step = 0
            score = 0
            continue


    df = pd.DataFrame(scores, columns=['Score'])
    df.to_csv(os.path.join(save_path, csv_episode_name), index=False)
    df = pd.DataFrame(highest_rewards, columns=['Highest Reward'])
    df.to_csv(os.path.join(save_path, csv_highest_reward_name), index=False)

    sys.exit()


