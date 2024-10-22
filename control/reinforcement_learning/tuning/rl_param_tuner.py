import numpy as np
import pandas as pd

from control.base_controller import BaseController
from control.reinforcement_learning.tuning.auto_tuner_params import AutoTunerParams
from state_space.inputs.control_action import ControlAction
from state_space.states.state import State
from vehicle.vehicle_info import VehicleInfo


class RLAutoTuner(BaseController):
    def __init__(
        self,
        params: AutoTunerParams,
        vi: VehicleInfo,
        controller: BaseController,
    ):
        super().__init__(params=params, vi=vi)
        self.current_reward = None
        self.controller = controller

        self.q_table = {}  # Q-table to store Q-values for state-action pairs

        # Define boundaries and discretized values for each parameter
        self.kp_values = np.linspace(0.5, 2.0, num=5)
        self.forward_gain_values = np.linspace(0.1, 0.5, num=5)
        self.look_ahead_distance_values = np.linspace(0.1, 0.5, num=25)

    def initialize(
        self,
        ref_df: pd.DataFrame,
        sampling_time: float,
        initial_state: State,
        starting_control_action: ControlAction,
    ):
        self.controller.initialize(
            ref_df, sampling_time, initial_state, starting_control_action
        )
        self.current_reward = None

    def reset(self):  # Reset the Controller to its initial state
        self.controller.reset()
        self.current_reward = None

    def store_transition(self, reward: float):
        self.current_reward = reward

    def update_policy(self):
        # Update Q-values using Q-learning update rule
        if self.current_reward is not None:
            current_state = self.params.sub_controller_params
            state_action = (current_state, action)
            if state_action not in self.q_table:
                self.q_table[
                    state_action
                ] = 0  # Initialize Q-value for new state-action pair
            self.q_table[state_action] += self.params.learning_rate * (
                self.current_reward
                + self.params.discount_factor * max(self.q_table.values())
                - self.q_table[state_action]
            )

    def select_action(self):
        # Action selection logic based on epsilon-greedy approach
        current_state = self.pure_pursuit_controller.current_state
        if np.random.rand() < self.params.epsilon:
            # Exploration: Randomly choose action
            kp = np.random.choice(self.kp_values)
            forward_gain = np.random.choice(self.forward_gain_values)
            look_ahead_distance = np.random.choice(self.look_ahead_distance_values)
        else:
            # Exploitation: Choose action with highest Q-value for the current state
            max_q_value = float("-inf")
            best_action = None
            for kp in self.kp_values:
                for forward_gain in self.forward_gain_values:
                    for look_ahead_distance in self.look_ahead_distance_values:
                        action = (kp, forward_gain, look_ahead_distance)
                        state_action = (current_state, action)
                        if state_action in self.q_table:
                            q_value = self.q_table[state_action]
                            if q_value > max_q_value:
                                max_q_value = q_value
                                best_action = action
            if best_action is not None:
                kp, forward_gain, look_ahead_distance = best_action
            else:
                # If Q-values are not available for the current state, choose a random action
                kp = np.random.choice(self.kp_values)
                forward_gain = np.random.choice(self.forward_gain_values)
                look_ahead_distance = np.random.choice(self.look_ahead_distance_values)
        return kp, forward_gain, look_ahead_distance

    def compute_action(self, index, vs):  # Current State of the vehicle
        self.update_policy()

        action = self.select_action()

        # Update parameters

        # Update the Pure Pursuit controller with new parameters
        self.pure_pursuit_controller.params = self.params.sub_controller_params

        # Reduce epsilon after each iteration to decrease exploration over time
        self.params.epsilon *= self.params.epsilon_decay

        # Run Pure Pursuit controller iteration with updated parameters
        return self.pure_pursuit_controller.compute_action(index, vs)