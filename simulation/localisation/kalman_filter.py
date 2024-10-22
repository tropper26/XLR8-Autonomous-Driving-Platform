from copy import deepcopy

import numpy as np

from state_space.inputs.control_action import ControlAction
from state_space.models.generic_model import GenericModel
from state_space.state_space import StateSpace
from state_space.states.state import State


class KalmanFilter:
    def __init__(self, model: GenericModel):
        self.model = model

        self.state_space = None
        self.sampling_time = None

        base_noise_scale = 0.1
        acceleration_variance = base_noise_scale
        steering_angle_variance = base_noise_scale
        gnss_variance = base_noise_scale

        self.R_gnss = np.diag([gnss_variance] * 2)  # gnss variance
        self.Q = np.diag(
            [acceleration_variance] * 1 + [steering_angle_variance] * 1
        )  # imu variance

        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])

        self.P_cov = np.zeros((6, 6))  # covariance of estimate

        self.kalman_gain = np.zeros((6, 2))

    def initialize(
        self,
        initial_state: State,
        starting_control_input: ControlAction,
        sampling_time: float,
    ):
        self.initial_state = initial_state.copy()
        self.starting_control_input = starting_control_input.copy()

        self.state_space = StateSpace(
            model=self.model,
            current_state=initial_state,
            last_input=starting_control_input,
            sampling_time=sampling_time,
        )
        self.sampling_time = sampling_time

    def reset(self):
        self.state_space = StateSpace(
            model=self.state_space.model,
            current_state=self.initial_state.copy(),
            last_input=self.starting_control_input.copy(),
            sampling_time=self.sampling_time,
        )
        self.P_cov = np.zeros((6, 6))
        self.kalman_gain = np.zeros((6, 2))

    def prediction_step(
        self,
        observed_longitudinal_a: float,
        observed_heading_angle: float,
        observed_steering_angle: float,
    ):
        new_U = ControlAction(d=observed_steering_angle, a=observed_longitudinal_a)

        self.state_space.update_control_input(new_U)

        new_X = State(
            X=self.state_space.X.X,
            Y=self.state_space.X.Y,
            Psi=observed_heading_angle,
            x_dot=self.state_space.X.x_dot,
            y_dot=self.state_space.X.y_dot,
            psi_dot=self.state_space.X.psi_dot,
        )

        self.state_space.update_based_on_obervations(new_X)

        self.state_space.propagate_model(self.sampling_time, 1)

        self.P_cov = (
            self.state_space.A @ self.P_cov @ self.state_space.A.T
            + self.state_space.B @ self.Q @ self.state_space.B.T
        )

        return deepcopy(self.state_space.X)

    def correction_step(self, observed_position_x, observed_position_y):
        Z = np.array([observed_position_x, observed_position_y]).reshape(2, 1)

        X_error = self.kalman_gain @ (
            Z - (self.H @ self.state_space.X.as_column_vector)
        )

        self.state_space.X.add_column_vector(X_error)

        self.P_cov = (np.eye(6) - self.kalman_gain @ self.H) @ self.P_cov

        self.kalman_gain = (
            self.P_cov
            @ self.H.T
            @ np.linalg.inv(self.H @ self.P_cov @ self.H.T + self.R_gnss)
        )

        return deepcopy(self.state_space.X)