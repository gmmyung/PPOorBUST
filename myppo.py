from ppo import Actor, Critic, Environment
import torch
from torch import Tensor, nn
from typing import Tuple


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
        self.__position = self.__position + self.__velocity * self.__time_step
        action = torch.clamp(action, min=-1.0, max=1.0)
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
