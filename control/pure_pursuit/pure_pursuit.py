import numpy as np

from control.base_controller import BaseController
from control.controller_viz_info import ControllerVizInfo, Types
from control.pure_pursuit.pure_pursuit_params import PurePursuitParams
from parametric_curves.trajectory import TrajectoryDiscretization
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
        trajectory_discretization: TrajectoryDiscretization,
    ) -> (ControlAction, ControllerVizInfo):
        steering_angle, target_X, target_Y = self.pure_pursuit_control(
            current_state, trajectory_discretization
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

    def pure_pursuit_control(
        self, current_state: State, trajectory_discretization: TrajectoryDiscretization
    ):
        rear_X, rear_Y = self.vi.rear_axle_position(current_state)

        target_index = int(trajectory_discretization.S.searchsorted(
            trajectory_discretization.S[0]
            + self.look_ahead_distance(current_state.x_dot)
        ))

        target_index = min(
            target_index, len(trajectory_discretization) - 1
        )  # Clamp target index to the last index if it exceeds

        target_X, target_Y = trajectory_discretization.X[target_index], trajectory_discretization.Y[target_index]

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