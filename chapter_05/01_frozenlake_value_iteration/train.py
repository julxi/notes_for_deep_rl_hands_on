# train.py

import gymnasium as gym
import numpy as np
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

from value_iteration_agent import ValueIterationAgent


# === Hyperparameters === #
EPSILON = 0.10
GAMMA = 0.99
EPOCHS = 5
TOTAL_EPISODES = 50_000
EXPLORATION_INTERVAL = 50
EVAL_INTERVAL = 100
REWARD_GOAL = 0.85


class Trainer:
    def __init__(
        self,
        env: gym.Env,
        agent: ValueIterationAgent,
        total_episodes: int,
        exploration_interval: int,
        eval_interval: int,
    ):
        self.env = env
        self.agent = agent
        self.total_episodes = total_episodes
        self.exploration_interval = exploration_interval
        self.eval_interval = eval_interval

    def _start_phase_tracking(self):
        self._is_exploring = True
        self._explore_count = 0
        self._eval_buffer = []
        self._eval_round = 0

    def _training_core(self, explore: bool) -> tuple[float, int]:
        """
        Core of the learning loop
        """
        reward, length = self.agent.run_episode(self.env, explore=explore)
        self.agent.perform_value_iteration()
        self.agent.update_policy()
        return reward, length

    def _record_and_schedule(
        self,
        episode: int,
        reward: float,
        length: int,
        best_mean: float,
        writer: SummaryWriter,
    ) -> float:
        """
        Handles logging and alternating between exploration and evaluation.
        """
        writer.add_scalar("Reward", reward, episode)
        writer.add_scalar("Length", length, episode)

        if self._is_exploring:
            self._explore_count += 1
            if self._explore_count >= self.exploration_interval:
                self._is_exploring = False
                self._explore_count = 0
        else:
            self._eval_buffer.append(reward)
            if len(self._eval_buffer) >= self.eval_interval:
                mean_r = np.mean(self._eval_buffer)
                writer.add_scalar("MeanReward", mean_r, self._eval_round)
                best_mean = max(best_mean, float(mean_r))
                self._eval_round += 1
                self._eval_buffer.clear()
                self._is_exploring = True

        return best_mean

    def train_epoch(self, writer: SummaryWriter) -> float:
        """
        Runs a full training epoch, alternating between explore and eval phases.
        Returns the best mean reward achieved in evaluation.
        """
        self._start_phase_tracking()
        best_mean = 0.0

        for ep in trange(self.total_episodes, desc="Episodes"):
            reward, length = self._training_core(explore=self._is_exploring)
            best_mean = self._record_and_schedule(ep, reward, length, best_mean, writer)

        return best_mean

    def train(self, epochs: int, log_tag: str = "-value-iteration"):
        """
        Orchestrates multi-epoch training. Each epoch gets a fresh entry in tensorboard.
        """
        for epoch in range(1, epochs + 1):
            print(f"\n=== Epoch {epoch}/{epochs} ===")
            writer = SummaryWriter(comment=log_tag)
            best_mean = self.train_epoch(writer)
            writer.close()

            solved = "YES" if best_mean >= REWARD_GOAL else "NO"
            print(f"Solved? {solved}  BestMean: {best_mean:.3f}")
            print(self.agent.policy_to_string(width=4, height=4))


def main():
    env = gym.make("FrozenLake-v1")
    agent = ValueIterationAgent(epsilon=EPSILON, gamma=GAMMA)

    trainer = Trainer(
        env=env,
        agent=agent,
        total_episodes=TOTAL_EPISODES,
        exploration_interval=EXPLORATION_INTERVAL,
        eval_interval=EVAL_INTERVAL,
    )
    trainer.train(epochs=EPOCHS)


if __name__ == "__main__":
    main()
