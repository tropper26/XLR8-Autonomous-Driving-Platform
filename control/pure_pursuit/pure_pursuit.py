import numpy as np
import pandas as pd

from control.base_controller import BaseController
from control.controller_viz_info import ControllerVizInfo, Types
from control.pure_pursuit.pure_pursuit_params import PurePursuitParams
from state_space.inputs.control_action import ControlAction
from state_space.states.state import State
from vehicle.vehicle_info import VehicleInfo


class PurePursuitController(BaseController):
    def __init__(
        self,
        params: PurePursuitParams,
        vi: VehicleInfo,
    ):
        super().__init__(params=params, vi=vi)
        self.params = params
        self.min_look_ahead_distance = 5  # m
        self.max_look_ahead_distance = 25  # m

    def look_ahead_distance(self, longitudinal_velocity: float) -> float:
        return max(
            self.min_look_ahead_distance,
            min(
                +self.params.forward_gain * longitudinal_velocity,
                self.max_look_ahead_distance,
            ),
        )

    def initialize(
        self,
        initial_state: State,
        starting_control_action: ControlAction,
    ):
        pass

    def compute_action(
        self,
        index,
        current_state: State,
        error_state: State,
        trajectory_df: pd.DataFrame,
    ) -> (ControlAction, ControllerVizInfo):
        steering_angle, target_X, target_Y = self.pure_pursuit_control(
            current_state, trajectory_df
        )

        steering_angle = np.clip(steering_angle, self.vi.min_d, self.vi.max_d)

        acceleration = self.proportional_control(error_state.x_dot)

        acceleration = np.clip(acceleration, self.vi.min_a, self.vi.max_a)

        return ControlAction(a=acceleration, d=steering_angle), ControllerVizInfo(
            viz_type=Types.Point,
            X=np.array([]),
            Y=np.array([]),
            ref_X=np.array([target_X]),
            ref_Y=np.array([target_Y]),
        )

    def proportional_control(self, velocity_error: float):
        return self.params.kp * velocity_error

    def pure_pursuit_control(self, current_state: State, trajectory_df: pd.DataFrame):
        rear_X, rear_Y = self.vi.rear_axle_position(current_state)

        current_arc_length = trajectory_df.iloc[0]["S"]
        target_index = np.searchsorted(
            trajectory_df.S.values,
            current_arc_length + self.look_ahead_distance(current_state.x_dot),
        )
        if target_index >= trajectory_df.shape[0]:
            target_index = trajectory_df.shape[0] - 1

        target_X = trajectory_df.iloc[target_index]["X"]
        target_Y = trajectory_df.iloc[target_index]["Y"]

        heading_error = (
            np.arctan2(target_Y - rear_Y, target_X - rear_X) - current_state.Psi
        )

        # Normalize the angle between -pi and pi
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        value = (
            2.0
            * self.vi.wheelbase
            * np.sin(heading_error)
            / self.look_ahead_distance(current_state.x_dot)
        )

        steering_angle = np.arctan2(value, 1)

        return steering_angle, target_X, target_Y