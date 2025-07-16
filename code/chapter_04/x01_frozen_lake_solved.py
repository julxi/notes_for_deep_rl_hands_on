"""
FrozenLake-v1 Policy Gradient Training using Cross-Entropy Method
This script trains a policy to solve the FrozenLake-v1 environment
using a neural network and the Cross-Entropy Method.
Core idea:
- We run some episodes using the current policy.
- We keep only the 'positive' ones (with returns > 0).
  We know that return = 1 are actually the elites.
- We train the policy to imitate the actions
  taken in those elite episodes.
- We use a low learning rate
  This should counter the stochasticity of the environment
"""

import gymnasium as gym
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

from cross_entropy_helpers import (
    Episode,
    generate_episodes,
    flatten_episodes,
)


# --------------------------- Hyperparameters ---------------------------- #
BATCH_SIZE = 100
LEARNING_RATE = 5e-4

# ------------------------- Benchmark Parameters ------------------------ #
EVALUATE_EVERY = 50
EVAL_SIZE = 2000
GOAL = 0.76  # Target average return (optimal policy has average return ~0.74)


# --------------------------- Environment Setup ------------------------- #
class DiscreteOneHotWrapper(gym.ObservationWrapper):
    """
    Wraps the environment's observations to be in one-hot encoded format.
    """

    def __init__(self, env: gym.Env):
        super(DiscreteOneHotWrapper, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        self.obs_shape = (int(env.observation_space.n),)
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, self.obs_shape, dtype=np.float32
        )

    def observation(self, observation):
        res = np.zeros(self.obs_shape, dtype=np.float32)
        res[observation] = 1.0
        return res


env = DiscreteOneHotWrapper(gym.make("FrozenLake-v1"))

# ----------------------------- Policy Network -------------------------- #
obs_size = 16
n_actions = 4
hidden_layer_size = 128

net = nn.Sequential(
    nn.Linear(obs_size, hidden_layer_size),
    nn.ReLU(),
    nn.Linear(hidden_layer_size, n_actions),
)


# ----------------------------- Logging and Evaluation -------------------------- #
def make_writer():
    """
    Creates a SummaryWriter that saves metrics to `/runs/chapter_04/frozen_lake`
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
    run_name = f"{timestamp}_lr{LEARNING_RATE}_batch{BATCH_SIZE}"
    log_dir = os.path.join("runs", "chapter_04", "frozen_lake", run_name)
    return SummaryWriter(log_dir=log_dir)


def log_metrics(prefix: str, batch: list[Episode], step, writer):
    returns = [ep.episode_return for ep in batch]
    lengths = [len(ep.states) for ep in batch]
    metrics = {
        "ret_mean": np.mean(returns),
        "ep_len_min": np.min(lengths),
        "ep_len_mean": np.mean(lengths),
        "ep_len_max": np.max(lengths),
    }
    for name, val in metrics.items():
        writer.add_scalar(f"{prefix}/{name}", val, step)
    return metrics["ret_mean"]


# --------------------------- Training Loop ---------------------------- #
print(net)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=net.parameters(), lr=LEARNING_RATE)
writer = make_writer()

it = 0
while True:
    it += 1

    # generate a batch of episodes
    batch = generate_episodes(BATCH_SIZE, env, net)
    log_metrics("Train", batch, it, writer)

    # filter episodes with positive return and, if any, do a gradient step
    positives = [ep for ep in batch if ep.episode_return > 0.0]
    if positives:
        states, actions = flatten_episodes(positives)
        optimizer.zero_grad()
        logits = net(states)
        loss = loss_fn(logits, actions)
        loss.backward()
        optimizer.step()
        writer.add_scalar("Loss/train", loss.item(), it)

    # evaluation using greedy action selection
    if it % EVALUATE_EVERY == 0:
        eval_batch = generate_episodes(EVAL_SIZE, env, net, greedy=True)
        mean_ret = log_metrics("Eval", eval_batch, it, writer)
        print(f"[Step {it:>7}] Eval mean return: {mean_ret:.3f}")
        if mean_ret >= GOAL:
            print(f"Solved in {it} steps! 🎉")
            writer.close()
            break
