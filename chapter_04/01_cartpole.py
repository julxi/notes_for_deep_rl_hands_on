import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Tuple
import gymnasium as gym
from dataclasses import dataclass
import torch.nn.functional as F
from torch.distributions import Categorical

from gymnasium.vector import AsyncVectorEnv

# Paramateres
BATCH_SIZE = 50
PERCENTILE = 70
MAX_EPOCHS = 0

# the environment
env = gym.make("CartPole-v1")

# the net
obs_size = 4
n_actions = 2
hidden_layer = 128

net = nn.Sequential(
    nn.Linear(obs_size, hidden_layer),
    nn.ReLU(),
    nn.Linear(hidden_layer, n_actions),
)
print(net)


@dataclass
class EpisodeStep:
    observation: np.ndarray
    action: int


@dataclass
class Episode:
    reward: float
    steps: List[EpisodeStep]


def net_action_probs(obs, net):
    obs_v = torch.tensor(obs, dtype=torch.float32)
    print(obs_v)
    logits = net(obs_v)  # shape: [1, n_actions]
    print(logits)
    probs = F.softmax(logits, dim=1)  # shape: [1, n_actions]
    return probs  # remove batch dimension, returns tensor of shape [n_actions]


print(net_action_probs([[1, 2, 3, 4]], net))


def create_batch(env, net, batch_size):
    batch = []

    while len(batch) < batch_size:
        is_done = False
        is_trunc = False
        episode_reward = 0.0
        episode_steps = []

        obs, _ = env.reset()

        sm = nn.Softmax(dim=1)
        while not (is_done or is_trunc):
            obs_v = torch.tensor(obs, dtype=torch.float32)
            logits = net(obs_v)
            probs = F.softmax(logits, dim=0)
            dist = Categorical(probs=probs)
            action = dist.sample().item()

            # take the action
            next_obs, reward, is_done, is_trunc, _ = env.step(action)

            # add everything to the thingy
            episode_reward += float(reward)
            step = EpisodeStep(observation=obs, action=action)
            episode_steps.append(step)

            # prepare next loop
            obs = next_obs

        batch.append(Episode(reward=episode_reward, steps=episode_steps))

    return batch


def filter_batch(
    batch: List[Episode], percentile: float
) -> Tuple[torch.FloatTensor, torch.LongTensor, float, float]:
    rewards = [s.reward for s in batch]
    reward_bound = float(np.percentile(rewards, percentile))
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for episode in batch:
        if episode.reward < reward_bound:
            continue
        train_obs.extend([step.observation for step in episode.steps])
        train_act.extend([step.action for step in episode.steps])

    train_obs_v = torch.FloatTensor(np.vstack(train_obs))
    train_act_v = torch.LongTensor(train_act)

    return train_obs_v, train_act_v, reward_bound, reward_mean


objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=net.parameters(), lr=0.01)

for epoch in range(MAX_EPOCHS):
    optimizer.zero_grad()

    obs_v, acts_v, reward_b, reward_m = filter_batch(
        create_batch(env, net, BATCH_SIZE), PERCENTILE
    )

    action_scores_v = net(obs_v)
    loss_v = objective(action_scores_v, acts_v)
    loss_v.backward()

    optimizer.step()

    print(f"{epoch}: loss={loss_v.item()}, reward_mean={reward_m}, rw_bound={reward_b}")

    if reward_m > 475:
        print("solved!")
        break
