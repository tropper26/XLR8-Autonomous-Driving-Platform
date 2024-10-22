import os
from abc import ABC, abstractmethod
from copy import deepcopy

import torch as T
import torch.nn as nn
import torch.nn.functional as F


class NetworkBase(ABC, T.nn.Module):
    def __init__(self, name: str, lr: float, layer_dims: tuple[int, int, int], norm_layers: bool = False):
        super().__init__()
        self.name = name
        self.norm_layers = norm_layers
        self.fc1 = nn.Linear(layer_dims[0], layer_dims[1])
        self.fc2 = nn.Linear(layer_dims[1], layer_dims[2])

        self.layers = [self.fc1, self.fc2]

        if norm_layers:
            self.bn1 = nn.LayerNorm(layer_dims[1])
            self.bn2 = nn.LayerNorm(layer_dims[2])

        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def init_weights_and_biases(self, values: list[float]):
        if len(values) != len(self.layers):
            raise ValueError('The number of values must be equal to the number of layers')

        for layer, value in zip(self.layers, values):
            nn.init.uniform_(layer.weight, -value, value)
            nn.init.uniform_(layer.bias, -value, value)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def save_checkpoint(self, directory):
        checkpoint_file = os.path.join(directory, f'{self.name}.pth')
        T.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_file)

    def load_checkpoint(self, directory):
        checkpoint_file = os.path.join(directory, f'{self.name}.pth')
        checkpoint = T.load(checkpoint_file)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f'Loaded checkpoint from {checkpoint_file}')

    def full_copy(self, copy_name: str):
        network_copy = deepcopy(self)
        network_copy.name = copy_name

        return network_copy

class CriticBase(NetworkBase):
    def forward(self, state, action):
        action_value = T.cat([state, action], dim=1)
        if self.norm_layers:
            action_value = F.relu(self.bn1(self.fc1(action_value)))
            action_value = F.relu(self.bn2(self.fc2(action_value)))
        else:
            action_value = F.relu(self.fc1(action_value))
            action_value = F.relu(self.fc2(action_value))

        return action_value

class LinearBase(NetworkBase):
    def forward(self, state: T.Tensor) -> T.Tensor:
        if self.norm_layers:
            f1 = F.relu(self.bn1(self.fc1(state)))
            f2 = F.relu(self.bn2(self.fc2(f1)))
        else:
            f1 = F.relu(self.fc1(state))
            f2 = F.relu(self.fc2(f1))
        return f2


class LinearTanhBase(NetworkBase):
    def forward(self, state: T.Tensor) -> T.Tensor:
        if self.norm_layers:
            f1 = T.tanh(self.bn1(self.fc1(state)))
            f2 = T.tanh(self.bn2(self.fc2(f1)))
        else:
            f1 = T.tanh(self.fc1(state))
            f2 = T.tanh(self.fc2(f1))
        return f2