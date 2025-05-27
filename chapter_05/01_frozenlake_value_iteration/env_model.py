from __future__ import annotations
from collections import Counter, defaultdict
from typing import DefaultDict, Dict, Tuple, Set

State = int
Action = int
SA = Tuple[State, Action]


class EnvModel:
    """
    Model of environment dynamics: maintains transition counts, probabilities, and reward estimates.
    """

    def __init__(self) -> None:
        self.actions: DefaultDict[State, Set[Action]] = defaultdict(set)
        self.reward_estimates: Dict[SA, float] = {}
        self.transition_estimates: Dict[SA, Dict[State, float]] = {}

        # Internal counts for update
        self._transition_counts: DefaultDict[SA, Counter[State]] = defaultdict(Counter)
        self._reward_sums: Dict[SA, float] = defaultdict(float)
        self._reward_counts: Counter[SA] = Counter()

    def update(
        self, state: State, action: Action, reward: float, next_state: State
    ) -> None:
        """
        Incorporate a new transition sample and update statistics.
        """
        sa = (state, action)

        # Update raw counts
        self._transition_counts[sa][next_state] += 1
        self._reward_sums[sa] += reward
        self._reward_counts[sa] += 1
        self.actions[state].add(action)

        # Recompute expected reward
        count = self._reward_counts[sa]
        self.reward_estimates[sa] = self._reward_sums[sa] / count

        # Recompute transition probabilities
        counts = self._transition_counts[sa]
        total = sum(counts.values())
        self.transition_estimates[sa] = {ns: cnt / total for ns, cnt in counts.items()}
