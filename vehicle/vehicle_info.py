import numpy as np

from state_space.states.augmented_state import AugmentedState
from state_space.states.state import State
from vehicle.static_vehicle_constraints import StaticVehicleConstraints
from vehicle.vehicle_params import VehicleParams


class VehicleInfo:
    """
    Constants related to the vehicle's dynamic properties.
    """

    def __init__(self, vp: VehicleParams, constraints: StaticVehicleConstraints):
        self.vp = vp
        self.static_constraints = constraints

    @property
    def wheelbase(self):
        return self.vp.lf + self.vp.lr

    @property
    def min_delta_d(self):
        return self.static_constraints.min_delta_d

    @property
    def max_delta_d(self):
        return self.static_constraints.max_delta_d

    @property
    def min_delta_a(self):
        return self.static_constraints.min_delta_a

    @property
    def max_delta_a(self):
        return self.static_constraints.max_delta_a

    @property
    def min_x_dot(self):
        return self.static_constraints.min_x_dot

    @property
    def max_x_dot(self):
        return self.static_constraints.max_x_dot

    @property
    def min_d(self):
        return self.static_constraints.min_d

    @property
    def max_d(self):
        return self.static_constraints.max_d

    @property
    def max_a(self):
        return self.static_constraints.max_x_dot_dot

    @property
    def min_a(self):
        return self.static_constraints.min_x_dot_dot

    def front_axle_position(self, state: State) -> (float, float):
        return self.vp.front_axle_position(state)

    def rear_axle_position(self, state: State) -> (float, float):
        return self.vp.rear_axle_position(state)

    def as_dictionary(self):
        return {
            "vp": self.vp.as_dictionary(),
            "static_constraints": self.static_constraints.as_dictionary(),
        }

    def __str__(self):
        return str(self.as_dictionary())

    __repr__ = __str__

    def compute_y_dot_min_max(self, x_dot: float) -> tuple[float, float]:
        """
        Calculate the minimum and maximum lateral velocity based on the current longitudinal velocity

        Parameters:
        - x_dot (float): The current longitudinal velocity of the vehicle

        Returns:
        - Tuple of two floats representing the minimum and maximum lateral velocities achievable by the vehicle
          based on the provided inputs.
        """
        min_y_dot = max(-0.17 * x_dot, self.static_constraints.min_y_dot_hard_limit)

        max_y_dot = min(0.17 * x_dot, self.static_constraints.max_y_dot_hard_limit)

        return min_y_dot, max_y_dot

    def compute_acceleration_min_max(
        self, state: AugmentedState
    ) -> tuple[float, float]:
        """
        Calculate the minimum and maximum acceleration based on the current state, vehicle parameters,
        and constraints set on maximum and minimum NET longitudinal acceleration.

        Parameters:
        - state (AugmentedState): Object containing the current state of the vehicle, including
          information like position, velocity, and orientation.

        Returns:
        - Tuple of two floats representing the minimum and maximum accelerations achievable by the vehicle
          based on the provided inputs.

        The 'get_min_max_acceleration' function calculates the minimum and maximum longitudinal accelerations
        that a vehicle can achieve under the given conditions. Longitudinal acceleration refers to the
        acceleration of the vehicle in the direction of its motion (forward or backward). This value is
        crucial for vehicle control and stability.

        The function performs the following steps:

        1. Calculate the lateral (side) force on the front tires (Fyf) using the tire slip angle,
           lateral velocity, and other vehicle parameters.

        2. Compute the longitudinal acceleration based on the lateral force (Fyf) and other factors.
           This acceleration is represented as 'x_dot_dot_to_a' and includes contributions from
           tire forces and gravitational effects.

        3. Add the minimum and maximum longitudinal acceleration constraints to 'x_dot_dot_to_a' to
           determine the achievable range of accelerations (a_min and a_max) under the given constraints.

        Finally, the function returns a tuple containing 'a_min' and 'a_max', representing the minimum
        and maximum achievable longitudinal accelerations, respectively.

        Note: The difference between acceleration and net longitudinal acceleration is that the latter
        includes the effects of gravity and tire forces, while the former does not. The net longitudinal
        acceleration is the acceleration that the vehicle experiences in the direction of its motion.
        So by constraining the net longitudinal acceleration, we are constraining the acceleration
        that the vehicle/passenger experiences in the direction of its motion.

        :param state: AugmentedState object containing the current state of the vehicle.
        :return: Tuple of two floats representing the minimum and maximum accelerations achievable by the vehicle.
        """
        u = 0.00001  # small value to avoid division by zero

        Fyf = self.vp.Cf * (
            state.previous_d
            - state.y_dot / (state.x_dot + u)
            - self.vp.lf * state.psi_dot / (state.x_dot + u)
        )

        x_dot_dot_to_a = (
            +(
                Fyf * np.sin(state.previous_d)
                + self.vp.mju * self.vp.m * VehicleParams.GRAVITY
            )
            / self.vp.m
            - state.psi_dot * state.y_dot
        )

        a_min = self.static_constraints.min_x_dot_dot + x_dot_dot_to_a

        a_max = self.static_constraints.max_x_dot_dot + x_dot_dot_to_a

        return a_min, a_max