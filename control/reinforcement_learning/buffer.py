import numpy as np
from scipy.sparse import csr_matrix, issparse


class ReplayBuffer:
    def __init__(self, max_size: int, input_shape: int, n_actions: int):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        state_: np.ndarray,
        done: bool,
    ):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(
        self, batch_size: int, mode="uniform"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        max_mem = min(self.mem_cntr, self.mem_size)

        if mode == "uniform":
            batch = np.random.choice(max_mem, batch_size)

            states = self.state_memory[batch]
            actions = self.action_memory[batch]
            rewards = self.reward_memory[batch]
            states_ = self.new_state_memory[batch]
            dones = self.terminal_memory[batch]

        elif mode == "all":
            states = self.state_memory[:max_mem]
            actions = self.action_memory[:max_mem]
            rewards = self.reward_memory[:max_mem]
            states_ = self.new_state_memory[:max_mem]
            dones = self.terminal_memory[:max_mem]
        else:
            raise ValueError(f"Mode {mode} not supported")

        return states, actions, rewards, states_, dones

    def save_to_disk(self, file_path: str):
        actual_size = min(self.mem_cntr, self.mem_size)

        save_dict = {
            "state_memory": csr_matrix(self.state_memory[:actual_size]),
            "new_state_memory": csr_matrix(self.new_state_memory[:actual_size]),
            "action_memory": csr_matrix(self.action_memory[:actual_size]),
            "reward_memory": csr_matrix(self.reward_memory[:actual_size]),
            "terminal_memory": csr_matrix(
                self.terminal_memory[:actual_size].astype(int)
            ),
        }
        print("Save dict", save_dict)
        np.savez_compressed(file_path, **save_dict)

    def load_from_disk(self, file_path: str):
        data = np.load(file_path, allow_pickle=True)
        for key in data:
            array = data[key].item()  # data[key] is np.ndarray, .item() converts to CSR

            if issparse(array):
                array = array.toarray()  # Convert sparse to dense
            max_index = min(max(array.shape), self.mem_size)

            match key:
                case "state_memory":
                    self.state_memory[:max_index] = array
                case "new_state_memory":
                    self.new_state_memory[:max_index] = array
                case "action_memory":
                    self.action_memory[:max_index] = array
                case "reward_memory":
                    array = array.flatten()  # Flatten to 1D
                    self.reward_memory[:max_index] = array
                case "terminal_memory":
                    array = array.flatten()
                    self.terminal_memory[:max_index] = array.astype(bool)

            self.mem_cntr = max(self.mem_cntr, max_index)

        print(f"Loaded {self.mem_cntr} samples from {file_path}")