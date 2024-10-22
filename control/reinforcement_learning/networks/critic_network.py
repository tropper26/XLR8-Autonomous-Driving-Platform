import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from control.reinforcement_learning.networks.network_base import CriticBase


class CriticNetwork(CriticBase):
    def __init__(
        self,
        name: str,
        lr: float,
        layer_dims: tuple[int, int, int],
        output_dim: int,
        norm_layers: bool = False,
    ):
        super(CriticNetwork, self).__init__(
            name=name, lr=lr, layer_dims=layer_dims, norm_layers=norm_layers
        )
        self.q1 = nn.Linear(layer_dims[2], output_dim)

        self.to(self.device)

    def forward(self, state: T.Tensor, action: T.Tensor) -> T.Tensor:
        action_value = super().forward(state, action)
        action_value = self.q1(action_value)

        return action_value


class DDPGCriticNetwork(CriticBase):
    def __init__(
        self,
        name: str,
        lr: float,
        layer_dims: tuple[int, int, int],
        output_dim: int,
        actions_dim: int,
        norm_layers: bool = False,
    ):
        super(DDPGCriticNetwork, self).__init__(
            name=name, lr=lr, layer_dims=layer_dims, norm_layers=norm_layers
        )
        self.action_value = nn.Linear(actions_dim, layer_dims[2])
        self.q = nn.Linear(layer_dims[2], output_dim)

        self.layers.append(self.action_value)
        self.layers.append(self.q)

        f1_weight = 1.0 / np.sqrt(self.fc1.weight.data.size()[0])
        f2_weight = 1.0 / np.sqrt(self.fc2.weight.data.size()[0])
        action_value_weight = 1.0 / np.sqrt(self.action_value.weight.data.size()[0])
        q_weight = 0.003

        self.init_weights_and_biases(
            [f1_weight, f2_weight, action_value_weight, q_weight]
        )

        self.to(self.device)

    def forward(self, state, action):
        state_value = F.relu(self.bn1(self.fc1(state)))
        state_value = self.bn2(self.fc2(state_value))

        action_value = self.action_value(action)

        state_action_value = T.add(state_value, action_value)
        state_action_value = F.relu(state_action_value)

        state_action_value = self.q(state_action_value)

        return state_action_value