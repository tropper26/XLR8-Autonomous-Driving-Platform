import numpy as np

from control.base_controller import BaseController
from control.controller_viz_info import ControllerVizInfo, Types
from control.stanley.stanley_params import StanleyParams
from parametric_curves.trajectory import TrajectoryDiscretization
from state_space.inputs.control_action import ControlAction
from state_space.states.state import State
from vehicle.vehicle_info import VehicleInfo


def cross(a, b):  # cross tooltip is bugged in numpy
    return np.cross(a, b)


class StanleyController(BaseController):
    def __init__(
        self,
        params: StanleyParams,
        vi: VehicleInfo,
    ):
        super().__init__(params=params, vi=vi)
        self.params = params

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
        cross_track_error = np.hypot(error_state.X, error_state.Y)

        sign = np.sign(error_state.Y)

        cross_track_error = sign * cross_track_error

        # Stanley Control Law
        steering_angle = error_state.Psi + np.arctan2(
            self.params.k * cross_track_error, current_state.x_dot
        )

        steering_angle = np.clip(steering_angle, self.vi.min_d, self.vi.max_d)

        # Proportional control for acceleration
        acceleration = self.proportional_control(error_state.x_dot)

        acceleration = np.clip(acceleration, self.vi.min_a, self.vi.max_a)

        return ControlAction(a=acceleration, d=steering_angle), ControllerVizInfo(
            viz_type=Types.Point,
            X=np.array([]),
            Y=np.array([]),
            ref_X=np.array([]),
            ref_Y=np.array([]),
        )

    def proportional_control(self, velocity_error: float):
        return self.params.kp * velocity_error