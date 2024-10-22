import os

import torch as T
from abc import ABC, abstractmethod

from torch import Tensor

from control.reinforcement_learning.buffer import ReplayBuffer
from control.reinforcement_learning.networks.network_base import NetworkBase


class BaseAgent(ABC):
    def __init__(
        self,
        state_dim: int,
        actions_dim: int,
        batch_size: int,
        gamma=0.99,
        tau=0.001,
        replay_buffer_max_size: int = 1000000,
    ):
        self.memory = ReplayBuffer(replay_buffer_max_size, state_dim, actions_dim)
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.networks: list[NetworkBase] = []
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(
        self, mode="uniform"
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        (states, actions, rewards, new_states, dones) = self.memory.sample_buffer(
            batch_size=self.batch_size, mode=mode
        )
        memory_tensors = (
            T.tensor(states, dtype=T.float).to(self.device),
            T.tensor(actions, dtype=T.float).to(self.device),
            T.tensor(rewards, dtype=T.float).to(self.device),
            T.tensor(new_states, dtype=T.float).to(self.device),
            T.tensor(dones).to(self.device),
        )
        return memory_tensors

    def save_models(self, base_dir: str, iteration: int):
        directory = os.path.join(base_dir, str(iteration))
        os.makedirs(directory, exist_ok=True)
        agent_data_file_path = os.path.join(directory, "agent_data.pth")
        memory_file_path = os.path.join(directory, "memory.npz")

        self.memory.save_to_disk(memory_file_path)

        T.save({"gamma": self.gamma, "tau": self.tau}, agent_data_file_path)

        for network in self.networks:
            network.save_checkpoint(directory)

    def load_models(self, base_dir: str, iteration: int = -1):
        directory = os.path.join(base_dir, str(iteration))
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory {directory} does not exist")

        agent_data_path = os.path.join(directory, "agent_data.pth")
        memory_file_path = os.path.join(directory, "memory.npz")

        self.memory.load_from_disk(memory_file_path)

        checkpoint = T.load(agent_data_path)
        self.gamma = checkpoint["gamma"]
        self.tau = checkpoint["tau"]

        for network in self.networks:
            network.load_checkpoint(directory)

        return iteration

    def update_network_parameters(self, src_network, dest_network, tau=None):
        if tau is None:
            tau = self.tau
        for param, target in zip(src_network.parameters(), dest_network.parameters()):
            target.data.copy_(tau * param.data + (1 - tau) * target.data)

    @abstractmethod
    def choose_action(self, observation):
        pass

    @abstractmethod
    def learn(self):
        pass