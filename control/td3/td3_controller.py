import numpy as np
import pandas as pd

from control.base_controller import BaseController
from control.td3.td3_agent import TD3Agent
from control.td3.td3_params import TD3Params
from state_space.inputs.control_action import ControlAction
from state_space.states.state import State
from vehicle.vehicle_info import VehicleInfo


class TD3Controller(BaseController):
    def __init__(self, params: TD3Params, vi: VehicleInfo):
        super().__init__(params=params, vi=vi)
        self.params = params
        self.current_reward = None
        max_action_bounds = [1, 1]
        min_action_bounds = [-1, -1]
        self.agent = TD3Agent(
            actor_lr=params.actor_lr,
            critic_lr=params.critic_lr,
            tau=params.tau,
            state_dim=6,
            actions_dim=2,
            update_actor_interval=params.update_actor_interval,
            warmup=params.warmup,
            max_action_bounds=max_action_bounds,
            min_action_bounds=min_action_bounds,
            hidden_layer_dims=params.hidden_layer_dims,
            gamma=params.gamma,
            batch_size=params.batch_size,
            replay_buffer_max_size=params.replay_buffer_max_size,
            noise=params.noise,
        )

    def initialize(
        self,
        ref_df: pd.DataFrame,
        sampling_time: float,
        initial_state: State,
        starting_control_action: ControlAction,
    ):
        self.initial_state = initial_state.copy()
        self.starting_control_input = starting_control_action.copy()
        self.previous_state = initial_state
        self.previous_control_input = starting_control_action

    def compute_action(self, index, current_state: State) -> ControlAction:
        self.agent.learn()

        action = self.agent.choose_action(current_state.as_column_vector.flatten())

        acceleration_percentage = action[0]
        steering_percentage = action[1]
        delta_acceleration = self.vi.max_delta_a * acceleration_percentage
        delta_steering = self.vi.max_delta_d * steering_percentage

        acceleration = np.clip(
            self.previous_control_input.a + delta_acceleration,
            self.vi.min_a,
            self.vi.max_a,
        )

        steering = np.clip(
            self.previous_control_input.d + delta_steering,
            self.vi.min_d,
            self.vi.max_d,
        )

        print(f"acceleration: {acceleration:.2f}, steering in degrees: {steering:.2f}")

        control_input = ControlAction(a=acceleration, d=np.radians(steering))
        self.previous_state = current_state
        self.previous_control_input = control_input
        return control_input

    def reset(self):
        self.previous_state = self.initial_state.copy()
        self.previous_control_input = self.starting_control_input.copy()

    def store_transition(
        self,
        state: State,
        control_action: ControlAction,
        reward: float,
        next_state: State,
        terminated: bool,
    ):
        self.agent.store_transition(
            state.as_column_vector.flatten(),
            control_action.as_column_vector.flatten(),
            reward,
            next_state.as_column_vector.flatten(),
            terminated,
        )