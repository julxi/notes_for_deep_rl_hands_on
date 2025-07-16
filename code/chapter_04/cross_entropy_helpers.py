import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from dataclasses import dataclass


@dataclass
class Episode:
    episode_return: float
    states: list[np.ndarray]
    actions: list[int]


def policy(net, state) -> torch.Tensor:
    """
    Compute the softmax action distribution for a single state.

    The output is a probability distribution (rank-1 tensor).
    """
    logits = net(torch.tensor(state))
    probs = F.softmax(logits, dim=0)
    return probs


def sample_action(state, net) -> int:
    """
    Sample an action from the policy's predicted distribution.
    """
    probs = policy(net, state)
    dist = Categorical(probs)
    action = dist.sample()
    return int(action.item())


def greedy_action(state, net) -> int:
    """
    Select the action with the highest predicted probability.
    """
    probs = policy(net, state)
    action = torch.argmax(probs)
    return int(action.item())


def generate_episodes(batch_size, env, net, greedy=False, gamma=1.0) -> list[Episode]:
    """
    Run multiple episodes using the policy network `net`.

    If `greedy=True`, this uses the most probable action
    at each step (used for evaluation).
    """
    batch = []
    for _ in range(batch_size):
        episode_return = 0.0
        states = []
        actions = []

        state, _ = env.reset()

        discounting = 1
        while True:
            # policy
            action = greedy_action(state, net) if greedy else sample_action(state, net)
            next_state, return_, is_done, is_trunc, _ = env.step(action)

            # update episode statistics
            episode_return += float(discounting * return_)
            states.append(state)
            actions.append(action)

            # prepare next loop
            if is_done or is_trunc:
                break
            state = next_state
            discounting = discounting * gamma

        batch.append(
            Episode(
                episode_return=episode_return,
                states=states,
                actions=actions,
            )
        )
    return batch


def filter_batch(
    batch: list[Episode], quantile_threshold: float
) -> tuple[list[Episode], float, float, float]:
    """
    Keep only the top-performing episodes (based on return)

    Returns:
        - Elite episodes
        - The return cutoff used to select elites.
        - The mean return across the whole batch (for logging).
        - the fraction of elites in the training data
    """
    returns = [ep.episode_return for ep in batch]
    cutoff_return = float(np.quantile(returns, quantile_threshold))
    mean_return = float(np.mean(returns))

    elites = [ep for ep in batch if ep.episode_return >= cutoff_return]

    fraction_elites = len(elites) / len(batch)

    return elites, cutoff_return, mean_return, fraction_elites


def flatten_episodes(
    batch: list[Episode],
) -> tuple[torch.FloatTensor, torch.LongTensor]:
    train_states = []
    train_actions = []
    for ep in batch:
        train_states.extend(ep.states)
        train_actions.extend(ep.actions)

    train_states_tensor = torch.FloatTensor(np.vstack(train_states))
    train_actions_tensor = torch.LongTensor(train_actions)

    return train_states_tensor, train_actions_tensor
