from typing import Dict, Tuple
import gymnasium as gym
from collections import defaultdict, Counter
from torch.utils.tensorboard.writer import SummaryWriter

ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
NUM_EPISODES = 100
DEFAULT_STATE_VALUE = 5.0

State = int
Action = int


class Model:
    def __init__(self):
        self.expected_rewards: Dict[Tuple[State, Action], float] = defaultdict(float)
        self.transition_probabilities: Dict[State, Dict[Action, Dict[State, float]]] = (
            {}
        )
        self.action_counts: Dict[Tuple[State, Action], int] = defaultdict(int)

    def update(
        self,
        state_initial: State,
        action: Action,
        reward: float,
        state_resulting: State,
    ):
        state_action = (state_initial, action)
        self.action_counts[state_action] += 1
        inc = 1 / self.action_counts[state_action]
        self.expected_rewards[state_action] += inc * (
            reward - self.expected_rewards[state_action]
        )
        tp = self.transition_probabilities[state_action]
        for state in tp:
            u = 0
            if state == state_resulting:
                u = 1
            tp[state] += inc * (u - tp[state])


class Agent:
    def __init__(self):
        self.values: Dict[State, float] = defaultdict(lambda: DEFAULT_STATE_VALUE)
        self.state_counts: Dict[State, int] = defaultdict(int)
        self.model = Model()

    def play_episode(self, env):
        total_reward = 0.0
        state, _ = env.reset()
        while True:
            action = self.select_action(state, env.action_space)
            new_state, reward, is_done, is_trunc, _ = env.step(action)
            self.model.update(state, action, reward, new_state)
            total_reward += reward
            if is_done or is_trunc:
                break
            state = new_state
        return total_reward

    def select_action(self, state: State, action_space: gym.spaces.Discrete):
        q = {}
        for action in range(action_space.n):
            q[action] = self.calc_action_value(state, action)
        return max(q, key=q.get)

    def calc_action_value(self, state: State, action: Action) -> float:
        return self.model.expected_rewards[(state, action)] + sum(
            [
                p * self.values[s2]
                for s2, p in self.model.transition_probabilities[state][action].items()
            ]
        )

    def value_iteration(self):
        for state in self.model.transition_probabilities:
            self.values[state] = max(
                [
                    self.calc_action_value(state, action)
                    for action in self.model.transition_probabilities[state]
                ]
            )


test_env = gym.make(ENV_NAME)
agent = Agent()
writer = SummaryWriter(comment="-value-iteration")
for i in range(NUM_EPISODES):

    reward = agent.play_episode(test_env)
    agent.value_iteration()
    print(f"{i}:{reward}")
    writer.add_scalar("reward", reward, i)

writer.close()
