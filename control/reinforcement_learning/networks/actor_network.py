import numpy as np
import torch as T
import torch.nn as nn

from control.reinforcement_learning.networks.network_base import LinearBase


class ActorNetwork(LinearBase):
    def __init__(
        self,
        name: str,
        lr: float,
        layer_dims: tuple[int, int, int],
        output_dim: int,
        norm_layers: bool = False,
    ):
        super(ActorNetwork, self).__init__(
            name=name, lr=lr, layer_dims=layer_dims, norm_layers=norm_layers
        )
        self.mu = nn.Linear(layer_dims[2], output_dim)
        self.layers.append(self.mu)

        self.to(self.device)

    def forward(self, state: T.Tensor) -> T.Tensor:
        prob = super().forward(state)
        prob = T.tanh(self.mu(prob))  # if action is > +/- 1 then multiply by max action

        return prob


class DDPGActorNetwork(LinearBase):
    def __init__(
        self,
        name: str,
        lr: float,
        layer_dims: tuple[int, int, int],
        output_dim: int,
        norm_layers: bool = False,
    ):
        super(DDPGActorNetwork, self).__init__(
            name=name, lr=lr, layer_dims=layer_dims, norm_layers=norm_layers
        )
        self.mu = nn.Linear(layer_dims[2], output_dim)
        self.layers.append(self.mu)

        f1_weight = 1.0 / np.sqrt(self.fc1.weight.data.size()[0])
        f2_weight = 1.0 / np.sqrt(self.fc2.weight.data.size()[0])
        mu_weight = 0.003

        self.init_weights_and_biases([f1_weight, f2_weight, mu_weight])

        self.to(self.device)

    def forward(self, state: T.Tensor) -> T.Tensor:
        prob = super().forward(state)
        prob = T.tanh(self.mu(prob))  # if action is > +/- 1 then multiply by max action

        return prob


class SACActorNetwork(LinearBase):
    def __init__(
        self,
        name: str,
        lr: float,
        layer_dims: tuple[int, int, int],
        output_dim: int,
        std_min,
        norm_layers: bool = False,
    ):
        super(SACActorNetwork, self).__init__(
            name=name, lr=lr, layer_dims=layer_dims, norm_layers=norm_layers
        )
        self.mu = nn.Linear(layer_dims[2], output_dim)
        self.sigma = nn.Linear(layer_dims[2], output_dim)
        self.std_min = std_min
        self.to(self.device)

    def forward(self, state: T.Tensor) -> tuple[T.Tensor, T.Tensor]:
        prob = super().forward(state)
        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = T.clamp(sigma, min=self.std_min, max=1)

        return mu, sigma