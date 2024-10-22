from torch import nn

from control.reinforcement_learning.networks.network_base import LinearBase


class ValueNetwork(LinearBase):
    def __init__(
        self, name: str, lr: float, layer_dims: tuple[int, int, int], output_dim: int
    ):
        super(ValueNetwork, self).__init__(name=name, lr=lr, layer_dims=layer_dims)
        self.v = nn.Linear(layer_dims[2], output_dim)

        self.to(self.device)

    def forward(self, state):
        prob = super().forward(state)
        return self.v(prob)