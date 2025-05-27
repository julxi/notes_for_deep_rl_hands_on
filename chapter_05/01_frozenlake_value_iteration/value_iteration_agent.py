from typing import Tuple, Set
import gymnasium as gym
import random
import numpy as np
from collections import defaultdict

from env_model import EnvModel, State, Action


class ValueIterationAgent:
    """
    Learns values via value-iteration over a learned model,
    and derives a greedy policy.
    """

    def __init__(self, epsilon: float, gamma: float):
        self.epsilon = epsilon
        self.gamma = gamma
        self.model = EnvModel()
        self.values = defaultdict(lambda: 0.0)
        self.policy = {}  # state -> best action

    def run_episode(self, env: gym.Env, explore: bool = False) -> Tuple[float, int]:
        s, _ = env.reset()
        total_reward = 0.0
        length = 0
        done = False

        while not done:
            a = self._choose_action(s, env, explore)
            s_next, r, done, _, _ = env.step(a)
            self.model.update(int(s), int(a), float(r), int(s_next))
            total_reward += float(r)
            length += 1
            s = s_next

        return total_reward, length

    def _choose_action(self, s: State, env: gym.Env, explore: bool) -> Action:
        # ε-greedy vs. greedy
        if explore and random.random() < self.epsilon:
            return env.action_space.sample()
        # if unseen state: fall back to random
        return self.policy.get(s, env.action_space.sample())

    def perform_value_iteration(self) -> None:
        """
        One sweep of Bellman optimality update over all known states.
        """
        for s, actions in self.model.actions.items():
            q_values = [self._action_value(s, a) for a in actions]
            self.values[s] = max(q_values)

    def update_policy(self) -> None:
        """
        Make policy greedy w.r.t. current value function.
        """
        new_policy = {}
        for s, actions in self.model.actions.items():
            best_a = max(actions, key=lambda a: self._action_value(s, a))
            new_policy[s] = best_a
        self.policy = new_policy

    def _action_value(self, s: State, a: Action) -> float:
        sa = (s, a)
        r_exp = self.model.reward_estimates[sa]
        trans = self.model.transition_estimates[sa]
        future = sum(p * self.values[s2] for s2, p in trans.items())
        return r_exp + self.gamma * future

    def policy_to_string(self, width: int, height: int) -> str:
        """
        Returns a grid representation of the current policy.
        """
        arrow = {0: "←", 1: "↓", 2: "→", 3: "↑"}
        grid = []
        for i in range(height):
            row = []
            for j in range(width):
                s = i * width + j
                a = self.policy.get(s, None)
                row.append(arrow[a] if a in arrow else "·")
            grid.append(" ".join(row))
        return "\n".join(grid)
