import torch
from torch import Tensor
from abc import ABC, abstractmethod
from typing import Tuple, List
import matplotlib.pyplot as plt


class Actor(ABC, torch.nn.Module):
    @abstractmethod
    def forward(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            state: [time_steps, actor_num, state_dim]
        Returns:
            action: [time_steps, actor_num, action_dim]
            log_prob: [time_steps, actor_num]
        """
        pass

    @abstractmethod
    def get_log_prob(self, state: Tensor, action: Tensor) -> Tensor:
        """
        Args:
            state: [time_steps, actor_num, state_dim]
            action: [time_steps, actor_num, action_dim]
        Returns:
            log_prob: [time_steps, actor_num]
        """
        pass

    @abstractmethod
    def get_state_dim(self) -> int:
        pass

    @abstractmethod
    def get_action_dim(self) -> int:
        pass

    @abstractmethod
    def get_actor_num(self) -> int:
        pass


class Critic(ABC, torch.nn.Module):
    @abstractmethod
    def get_state_dim(self) -> int:
        pass

    @abstractmethod
    def forward(self, state: Tensor) -> Tensor:
        """
        Args:
            state: [time_steps, actor_num, state_dim]
        Returns:
            value: [time_steps, actor_num]
        """
        pass


class Environment(ABC):
    @abstractmethod
    def reset(self) -> Tensor:
        """
        Returns:
            state: [actor_num, state_dim]
        """
        pass

    @abstractmethod
    def step(self, action: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            action: [actor_num, action_dim]
        Returns:
            state: [actor_num, state_dim]
            reward: [actor_num]
        """
        pass

    @abstractmethod
    def get_state_dim(self) -> int:
        pass

    @abstractmethod
    def get_action_dim(self) -> int:
        pass

    @abstractmethod
    def get_actor_num(self) -> int:
        pass


class Storage:
    def __init__(
        self, time_steps: int, actor_num: int, state_dim: int, action_dim: int
    ) -> None:
        self.__actor_num = actor_num
        self.__time_steps = time_steps
        self.__state_dim = state_dim
        self.__action_dim = action_dim
        self.__states = torch.zeros(time_steps, actor_num, state_dim)
        self.__actions = torch.zeros(time_steps, actor_num, action_dim)
        self.__rewards = torch.zeros(time_steps, actor_num)
        self.__log_probs = torch.zeros(time_steps, actor_num)
        self.__time = 0

    def clear(self) -> None:
        # TODO: don't zero the data, just clear the grads.
        self.__states = torch.zeros(
            self.__time_steps, self.__actor_num, self.__state_dim
        )
        self.__actions = torch.zeros(
            self.__time_steps, self.__actor_num, self.__action_dim
        )
        self.__rewards = torch.zeros(self.__time_steps, self.__actor_num)
        self.__log_probs = torch.zeros(self.__time_steps, self.__actor_num)
        self.__time = 0

    def add(
        self, state: Tensor, action: Tensor, reward: Tensor, log_prob: Tensor
    ) -> None:
        """
        Args:
            state: [actor_num, state_dim]
            action: [actor_num, action_dim]
            reward: [actor_num]
            log_prob: [actor_num]
        """
        assert self.__time < self.__time_steps
        assert state.shape == (self.__actor_num, self.__state_dim)
        assert action.shape == (self.__actor_num, self.__action_dim)
        assert reward.shape == (self.__actor_num,)
        assert log_prob.shape == (self.__actor_num,)

        self.__states[self.__time] = state
        self.__actions[self.__time] = action
        self.__rewards[self.__time] = reward
        self.__log_probs[self.__time] = log_prob

        self.__time += 1

    def get(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Returns:
            states: [time_steps, actor_num, state_dim]
            actions: [time_steps, actor_num, action_dim]
            rewards: [time_steps, actor_num]
            log_probs: [time_steps, actor_num, action_dim]
        """
        assert self.__time == self.__time_steps
        return self.__states, self.__actions, self.__rewards, self.__log_probs


class PPO:
    def __init__(
        self,
        actor: Actor,
        critic: Critic,
        env: Environment,
        clip_ratio: float,
        time_steps_per_epoch: int,
        discount_factor: float,
        actor_training_epochs: int,
        critic_training_epochs: int,
        actor_batch_size: int,
        critic_batch_size: int,
    ) -> None:
        assert actor.get_state_dim() == env.get_state_dim() == critic.get_state_dim()
        assert actor.get_action_dim() == env.get_action_dim()
        assert time_steps_per_epoch % actor_batch_size == 0
        assert time_steps_per_epoch % critic_batch_size == 0
        self.__actor = actor
        self.__critic = critic
        self.__env = env
        self.__state_dim = env.get_state_dim()
        self.__action_dim = env.get_action_dim()
        self.__actor_num = env.get_actor_num()
        self.__discount_factor = discount_factor
        self.__time_steps_per_epoch = time_steps_per_epoch
        self.__actor_optimizer = torch.optim.Adam(self.__actor.parameters(), lr=1e-4)
        self.__critic_optimizer = torch.optim.Adam(self.__critic.parameters(), lr=1e-4)
        self.__actor_training_epochs = actor_training_epochs
        self.__critic_training_epochs = critic_training_epochs
        self.__actor_batch_size = actor_batch_size
        self.__critic_batch_size = critic_batch_size
        self.__clip_ratio = clip_ratio
        self.__storage = Storage(
            time_steps_per_epoch, self.__actor_num, self.__state_dim, self.__action_dim
        )
        self.__reward_to_go_multiplier = self.reward_to_go_multiplier()

    def reward_to_go_multiplier(self) -> Tensor:
        """Calculates reward-to-go multiplier matrix.
        e.g. size=5, discount_factor=0.9
        [[1, 0, 0, 0, 0],
        [0.9, 1, 0, 0, 0],
        [0.81, 0.9, 1, 0, 0],
        [0.729, 0.81, 0.9, 1, 0],
        [0.6561, 0.729, 0.81, 0.9, 1]]
        """
        size = self.__time_steps_per_epoch
        row_indices = torch.arange(size).unsqueeze(1).expand(size, size)
        col_indices = torch.arange(size).flip(0).unsqueeze(0).expand(size, size)
        sum_matrix = row_indices + col_indices

        return (
            (self.__discount_factor ** (sum_matrix.sub(size - 2).relu() - 1))
            .where(sum_matrix > size - 2, 0.0)
            .repeat(self.__actor_num, 1, 1)
        )

    def reward_to_go(self, rewards: Tensor) -> Tensor:
        """Calculates rewards-to-go.
        Args:
            rewards: [time_steps, actor_num]
        Returns:
            rewards_to_go: [time_steps, actor_num]
        """
        assert rewards.shape == (self.__time_steps_per_epoch, self.__actor_num)

        return (
            rewards.transpose(0, 1)
            .reshape(self.__actor_num, -1, self.__time_steps_per_epoch)
            .bmm(self.__reward_to_go_multiplier)
            .reshape(self.__actor_num, self.__time_steps_per_epoch)
            .transpose(0, 1)
        )

    def train(self) -> float:
        """
        Returns:
            mean_reward: float
        """
        self.__storage.clear()
        state = self.__env.reset()
        for _ in range(self.__time_steps_per_epoch):
            action, log_prob = self.__actor.forward(state.unsqueeze(0))
            state, reward = self.__env.step(action.squeeze(0))
            self.__storage.add(state, action.squeeze(0), reward, log_prob.squeeze(0))

        states, actions, rewards, log_probs = self.__storage.get()

        rewards_to_go = self.reward_to_go(rewards)

        advantages = rewards_to_go - self.__critic.forward(states).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.train_actor(states, actions, log_probs, advantages)
        self.train_critic(states, rewards_to_go)

        return torch.mean(rewards).item()

    def train_actor(
        self, states: Tensor, actions: Tensor, log_probs: Tensor, advantages: Tensor
    ) -> None:
        """
        Args:
            states: [time_steps, actor_num, state_dim]
            actions: [time_steps, actor_num, action_dim]
            log_probs: [time_steps, actor_num]
            advantages: [time_steps, actor_num]
        """
        self.__actor.train()
        for _ in range(self.__actor_training_epochs):
            indices = torch.randperm(self.__time_steps_per_epoch)
            states = states[indices].detach()
            actions = actions[indices].detach()
            log_probs = log_probs[indices].detach()
            advantages = advantages[indices].detach()
            for start in range(0, states.shape[0], self.__actor_batch_size):
                self.__actor_optimizer.zero_grad()
                batch_states = states[start : start + self.__actor_batch_size]
                batch_actions = actions[start : start + self.__actor_batch_size]
                batch_log_probs = log_probs[start : start + self.__actor_batch_size]
                batch_advantages = advantages[start : start + self.__actor_batch_size]
                new_log_probs = self.__actor.get_log_prob(batch_states, batch_actions)
                ratio = torch.exp(new_log_probs - batch_log_probs)
                surr_1 = ratio * batch_advantages
                surr_2 = (
                    torch.clamp(ratio, 1.0 - self.__clip_ratio, 1.0 + self.__clip_ratio)
                    * batch_advantages
                )
                surr_loss = -torch.min(surr_1, surr_2).mean()
                print("surr_loss", surr_loss.item())
                surr_loss.backward()
                self.__actor_optimizer.step()

    def train_critic(self, states: Tensor, rewards_to_go: Tensor) -> None:
        """
        Args:
            states: [time_steps, actor_num, state_dim]
            rewards_to_go: [time_steps, actor_num]
        """
        self.__critic.train()
        for _ in range(self.__critic_training_epochs):
            indices = torch.randperm(self.__time_steps_per_epoch)
            states = states[indices]
            rewards_to_go = rewards_to_go[indices]
            total_loss = 0
            for start in range(0, states.shape[0], self.__critic_batch_size):
                self.__critic_optimizer.zero_grad()
                batch_states = states[start : start + self.__critic_batch_size]
                batch_rewards_to_go = rewards_to_go[
                    start : start + self.__critic_batch_size
                ]
                loss = torch.mean(
                    torch.square(
                        self.__critic.forward(batch_states) - batch_rewards_to_go
                    )
                )
                loss.backward()
                total_loss += loss
                self.__critic_optimizer.step()
            print("mean_loss", total_loss / self.__critic_training_epochs)


import torch.nn as nn


class MyActor(Actor):
    def __init__(self, actor_dim: int, state_dim: int, actor_num: int) -> None:
        # simple normal distribution
        super().__init__()
        self.__state_dim = state_dim
        self.__actor_dim = actor_dim
        self.__actor_num = actor_num
        self.__network = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.__mu = nn.Linear(32, actor_dim)
        self.__sigma = nn.Linear(32, actor_dim)

    def forward(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.__network(state)
        mu = self.__mu(x)
        sigma = torch.exp(self.__sigma(x))
        # check if sigma is positive
        action = torch.distributions.Normal(mu, sigma).sample()
        log_prob = torch.distributions.Normal(mu, sigma).log_prob(action).sum(dim=2)
        return action, log_prob

    def get_log_prob(self, state: Tensor, action: Tensor) -> Tensor:
        x = self.__network(state)
        mu = self.__mu(x)
        sigma = torch.exp(self.__sigma(x))
        return torch.distributions.Normal(mu, sigma).log_prob(action).sum(dim=2)

    def get_state_dim(self) -> int:
        return self.__state_dim

    def get_action_dim(self) -> int:
        return self.__actor_dim

    def get_actor_num(self) -> int:
        return self.__actor_num


class MyCritic(Critic):
    def __init__(
        self, state_dim: int, actor_num: int, mini_batch_size: int, training_epochs: int
    ) -> None:
        super().__init__()
        self.__state_dim = state_dim
        self.__actor_num = actor_num
        self.__mini_batch_size = mini_batch_size
        self.__training_epochs = training_epochs
        self.__network = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.__network(state).squeeze(2)

    def get_state_dim(self) -> int:
        return self.__state_dim


# Super simple environment, just have to move in 2d space
class MyEnvironment(Environment):
    def __init__(self, actor_num: int, time_step: float, max_vel=0.5) -> None:
        self.__actor_num = actor_num
        self.__position = torch.zeros(actor_num, 2)
        self.__velocity = torch.zeros(actor_num, 2)
        self.__target = torch.zeros(actor_num, 2)
        self.__time_step = time_step
        self.__max_vel = max_vel

    def __get_state(self) -> Tensor:
        return torch.cat([self.__position, self.__velocity, self.__target], 1)

    def get_reward(self) -> Tensor:
        dist = torch.norm(self.__position - self.__target, dim=1)
        return torch.clamp(dist.reciprocal() - 1, 0, 1).detach()

    def reset(self) -> Tensor:
        self.__position = torch.rand(self.__actor_num, 2).mul(2).sub(1)
        self.__target = torch.rand(self.__actor_num, 2).mul(2).sub(1)
        self.__velocity = torch.rand(self.__actor_num, 2).sub(0.5) / 10
        return self.__get_state().detach()

    def step(self, action: Tensor) -> Tuple[Tensor, Tensor]:
        self.__position = self.__position + self.__velocity
        self.__velocity = torch.clamp(
            self.__velocity + action * self.__time_step,
            min=-self.__max_vel,
            max=self.__max_vel,
        )
        return self.__get_state(), self.get_reward()

    def get_state_dim(self) -> int:
        return 6

    def get_action_dim(self) -> int:
        return 2

    def get_actor_num(self) -> int:
        return self.__actor_num


torch.autograd.set_detect_anomaly(True)
env = MyEnvironment(256, 1 / 60)
act = MyActor(env.get_action_dim(), env.get_state_dim(), env.get_actor_num())
crit = MyCritic(env.get_state_dim(), env.get_actor_num, 32, 4)
ppo = PPO(
    act,
    crit,
    env,
    clip_ratio=0.1,
    time_steps_per_epoch=128,
    discount_factor=0.9,
    actor_training_epochs=32,
    critic_training_epochs=32,
    actor_batch_size=32,
    critic_batch_size=32,
)

for _ in range(100):
    print("rewards", ppo.train())


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.lines as lines

import numpy as np

# 4x4 subplots
fig, ax = plt.subplots(4, 8)
assert isinstance(ax, np.ndarray)
state = env.reset()
action, _ = act.forward(state.unsqueeze(0))
states, rewards = env.step(action.squeeze(0))
actor_markers: List[patches.Circle] = []
target_markers: List[patches.Circle] = []
velocity_lines: List[lines.Line2D] = []
reward_bars: List[patches.Rectangle] = []

for i in range(16):
    map_subplot = ax[i // 4][(i % 4) * 2]
    map_subplot.clear()
    actor_markers.append(
        patches.Circle(
            (states[i][0].item(), states[i][1].item()), 0.1, color="r", zorder=1
        )
    )
    target_markers.append(
        patches.Circle(
            (states[i][4].item(), states[i][5].item()), 0.1, color="g", zorder=2
        )
    )
    velocity_lines.append(
        map_subplot.plot(
            [states[i][0].item(), states[i][0].item() + states[i][2].item() * 10],
            [states[i][1].item(), states[i][1].item() + states[i][3].item() * 10],
            linewidth=1,
            color="b",
            zorder=0,
        )[0]
    )

    map_subplot.set_xlim(-1.5, 1.5)
    map_subplot.set_ylim(-1.5, 1.5)
    map_subplot.add_patch(actor_markers[i])
    map_subplot.add_patch(target_markers[i])
    map_subplot.set_aspect("equal")
    map_subplot.get_xaxis().set_visible(False)
    map_subplot.get_yaxis().set_visible(False)

    rewards_subplot = ax[i // 4][(i % 4) * 2 + 1]
    rewards_subplot.set_xlim(0, 0.2)
    rewards_subplot.set_ylim(0, 1)

    reward_bars.append(
        patches.Rectangle(
            (0, 0),
            rewards_subplot.get_xlim()[1],
            rewards_subplot.get_ylim()[1] * rewards[i],
        )
    )
    rewards_subplot.add_patch(reward_bars[i])
    rewards_subplot.set_aspect("equal")
    rewards_subplot.get_xaxis().set_visible(False)
    rewards_subplot.get_yaxis().set_visible(False)


def update(frame):
    action = act.forward(state.unsqueeze(0))[0].squeeze(0)
    states, rewards = env.step(action)
    for i in range(16):
        actor_markers[i].center = (states[i][0].item(), states[i][1].item())
        target_markers[i].center = (states[i][4].item(), states[i][5].item())

        velocity_lines[i].set_data(
            [states[i][0].item(), states[i][0].item() + states[i][2].item() * 10],
            [states[i][1].item(), states[i][1].item() + states[i][3].item() * 10],
        )

        reward_bars[i].set_height(rewards_subplot.get_ylim()[1] * rewards[i])

    return actor_markers + target_markers + velocity_lines


anim = animation.FuncAnimation(fig, update, 5000, interval=1000 / 60)


plt.show()
