import numpy as np

from dto.form_dto import FormDTO
from state_space.states.state import State


class VehicleParams(FormDTO):
    """
    Relevant information about the vehicle.
    """

    GRAVITY = 9.81  # m/s^2

    def __init__(
        self,
        mass: float,  # kg
        moment_of_inertia: float,  # kg m^2
        front_tire_stiffness_coefficient: float,  # N/rad
        rear_tire_stiffness_coefficient: float,  # N/rad
        front_axle_length: float,  # distance from center of mass to front axle
        rear_axle_length: float,  # distance from center of mass to rear axle
        friction_coefficient: float,  # friction coefficient between tires and road material
    ):
        self.m = mass
        self.Iz = moment_of_inertia
        self.Cf = front_tire_stiffness_coefficient
        self.Cr = rear_tire_stiffness_coefficient
        self.lf = front_axle_length
        self.lr = rear_axle_length
        self.mju = friction_coefficient
        self.width = 2  # 0.25  # 1.5  # m
        self.wheel_width = 0.2
        self.wheel_diameter = 0.5

    def is_bicycle(self) -> bool:
        return self.width <= 0.25

    @property
    def wheelbase(self):
        return self.lf + self.lr

    def attributes_to_ignore(self) -> list[str]:
        return []

    def front_axle_position(self, state: State) -> (float, float):
        front_axle_X = state.X + self.lf * np.cos(state.Psi)
        front_axle_Y = state.Y + self.lf * np.sin(state.Psi)
        return front_axle_X, front_axle_Y

    def rear_axle_position(self, state: State) -> (float, float):
        rear_wheel_X = state.X - self.lr * np.cos(state.Psi)
        rear_wheel_Y = state.Y - self.lr * np.sin(state.Psi)
        return rear_wheel_X, rear_wheel_Y

    def convert_x_dot_dot_to_a(
        self, x_dot_dot: float, state: State, steering_angle: float
    ) -> float:
        Fyf = self.Cf * (
            steering_angle
            - state.y_dot / state.x_dot
            - self.lf * state.psi_dot / state.x_dot
        )

        result = (
            x_dot_dot
            + (Fyf * np.sin(steering_angle) + self.mju * self.m * VehicleParams.GRAVITY)
            / self.m
            - state.psi_dot * state.y_dot
        )[
            0
        ]  # just because of formatting, it makes a tuple with one element

        return result

    def as_dictionary(self):
        """
        Convert the object's attributes to a dictionary.
        """
        return {
            "mass": self.m,
            "moment_of_inertia": self.Iz,
            "front_tire_stiffness_coefficient": self.Cf,
            "rear_tire_stiffness_coefficient": self.Cr,
            "front_axle_length": self.lf,
            "rear_axle_length": self.lr,
            "friction_coefficient": self.mju,
        }

    @classmethod
    def from_dict(cls, vehicle_info_dict):
        return cls(
            vehicle_info_dict["mass"],
            vehicle_info_dict["moment_of_inertia"],
            vehicle_info_dict["front_tire_stiffness_coefficient"],
            vehicle_info_dict["rear_tire_stiffness_coefficient"],
            vehicle_info_dict["front_axle_length"],
            vehicle_info_dict["rear_axle_length"],
            vehicle_info_dict["friction_coefficient"],
        )

    def __str__(self):
        return f"VehicleInfo: {self.as_dictionary()}"

    __repr__ = __str__