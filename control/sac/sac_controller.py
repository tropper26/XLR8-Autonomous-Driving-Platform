import numpy as np
import pandas as pd

from control.base_controller import BaseController
from control.controller_viz_info import ControllerVizInfo
from control.sac.sac_agent import SACAgent
from control.sac.sac_params import SACParams
from state_space.inputs.control_action import ControlAction
from state_space.states.state import State
from vehicle.vehicle_info import VehicleInfo


class SACController(BaseController):
    def __init__(self, params: SACParams, vi: VehicleInfo):
        super().__init__(params=params, vi=vi)
        self.previous_agent_action = None
        self.previous_agent_state = None
        self.starting_control_input = None
        self.full_agent_state = None
        self.params = params
        self.current_reward = None
        max_action_bounds = [1, 1]
        self.agent = SACAgent(
            actor_lr=params.actor_lr,
            critic_lr=params.critic_lr,
            tau=params.tau,
            state_dim=10,
            actions_dim=2,
            max_action_bounds=max_action_bounds,
            hidden_layer_dims=params.hidden_layer_dims,
            gamma=params.gamma,
            batch_size=params.batch_size,
            replay_buffer_max_size=params.replay_buffer_max_size,
            reward_scale=params.reward_scale,
        )

    def initialize(
        self,
        initial_state: State,
        starting_control_action: ControlAction,
    ):
        self.previous_control_action = starting_control_action.copy()
        self.previous_agent_action = np.array([0, 0])
        self.previous_agent_state = np.array(
            [
                0,
                0,
                0,
                0,
                0,
            ]
        )
        self.previous_full_agent_state = np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        )
        self.counter = 0

    def compute_action(
        self,
        index,
        current_state: State,
        error_state: State,
        trajectory_discretization: pd.DataFrame,
    ) -> (ControlAction, ControllerVizInfo):
        self.counter += 1
        if self.counter % 4 == 0:
            print("Learning")
            self.agent.learn()

        current_agent_state = np.array(
            [
                error_state.Y,
                error_state.Psi,
                error_state.x_dot,
                self.previous_control_action.d,
                trajectory_discretization.iloc[0].K,
            ]
        )

        self.full_agent_state = np.concatenate(
            (current_agent_state, self.previous_agent_state)
        )

        action = self.agent.choose_action(self.full_agent_state)

        acceleration_percentage = action[0]
        steering_percentage = action[1]

        delta_acceleration = self.vi.max_delta_a * acceleration_percentage
        delta_steering = self.vi.max_delta_d * steering_percentage

        acceleration = np.clip(
            self.previous_control_action.a + delta_acceleration,
            self.vi.min_a,
            self.vi.max_a,
        )

        steering = np.clip(
            self.previous_control_action.d + delta_steering,
            self.vi.min_d,
            self.vi.max_d,
        )

        control_input = ControlAction(a=acceleration, d=steering)
        self.previous_agent_action = action.copy()
        return control_input, None

    def store_transition(
        self,
        state: State,
        control_action: ControlAction,
        reward: float,
        next_state: State,
        terminated: bool,
    ):
        self.previous_control_action = control_action.copy()
        self.agent.store_transition(
            state=self.previous_full_agent_state,
            action=self.previous_agent_action,
            reward=reward,
            state_=self.full_agent_state,
            done=terminated,
        )
        self.previous_agent_state = self.full_agent_state[
            :5
        ].copy()  # The first 5 elements are the new last state
        self.previous_full_agent_state = self.full_agent_state.copy()