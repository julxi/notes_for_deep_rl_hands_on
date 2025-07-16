"""
CartPole-v1 Policy Gradient Training using Cross-Entropy Method

This script trains a policy to solve the CartPole-v1 environment
using a neural network and the Cross-Entropy Method.

Core idea:
- We run many episodes using the current policy.
- We keep only the 'elite' ones (with highest returns).
- We train the policy to imitate the actions
  taken in those elite episodes.
"""

import numpy as np
import gymnasium as gym
import torch.nn as nn
import torch.optim as optim

from cross_entropy_helpers import (
    generate_episodes,
    filter_batch,
    flatten_episodes,
)


# --------------------------- Hyperparameters ---------------------------- #
BATCH_SIZE = 10  # Sample size per training iteration
QUANTILE_THRESHOLD = 0.9  # Top fraction of sample's episodes to use for training
LEARNING_RATE = 5e-3

# ------------------------- Evaluation Parameters ------------------------ #
EVALUATE_EVERY = 50  # Evaluation frequency (iterations)
EVAL_SIZE = 250  # Number of episodes for evaluation
GOAL = 500.0  # Target average return (CartPole-v1 max is 500)

# --------------------------- Environment Setup ------------------------- #
env = gym.make("CartPole-v1")

# ----------------------------- Policy Network -------------------------- #
obs_size = 4
n_actions = 2
hidden_layer_size = 128

net = nn.Sequential(
    nn.Linear(obs_size, hidden_layer_size),
    nn.ReLU(),
    nn.Linear(hidden_layer_size, n_actions),
)


# ----------------------------- Task Specific Evaluation -------------------------- #
def evaluate(net, env, n_episodes):
    """
    Evaluate policy network `net` (greedily) for a number of episodes.
    """
    batch = generate_episodes(n_episodes, env, net, greedy=True)
    returns = [ep.episode_return for ep in batch]
    min_ret = np.min(returns)
    mean_ret = np.mean(returns)
    max_ret = np.max(returns)

    return min_ret, mean_ret, max_ret


# --------------------------- Training Loop ---------------------------- #
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=net.parameters(), lr=LEARNING_RATE)

print(net)

it = 0
while True:
    it += 1

    # generate training data
    batch = generate_episodes(BATCH_SIZE, env, net)
    elites, return_cutoff, return_mean, fraction_elites = filter_batch(
        batch, QUANTILE_THRESHOLD
    )
    train_states, target_actions = flatten_episodes(elites)

    # train
    optimizer.zero_grad()
    action_logits = net(train_states)
    loss = loss_fn(action_logits, target_actions)
    loss.backward()
    optimizer.step()

    # log
    print(
        f"Iteration {it:4d} — "
        f"loss: {loss.item():.4f} | "
        f"mean_return: {return_mean:6.2f} | "
        f"{QUANTILE_THRESHOLD}th percentile cutoff: {return_cutoff:6.2f} | "
        f"{fraction_elites*100:5.1f}% are elites"
    )

    # evaluate
    if it % EVALUATE_EVERY == 0:
        min_r, mean_r, max_r = evaluate(net, env, EVAL_SIZE)
        print(f">>> Benchmark: min {min_r}, mean {mean_r:.1f}, max {max_r}")
        if mean_r >= GOAL:
            print(f"Solved after {it} iterations! 🚀")
            break
