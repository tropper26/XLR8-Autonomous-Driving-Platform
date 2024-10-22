import numpy as np

from dto.form_dto import FormDTO


class StaticVehicleConstraints(FormDTO):
    """
    Constants related to the vehicle's static properties.
    """

    def __init__(
        self,
        min_x_dot: float,  # minimum value of the longitudinal velocity m/s
        max_x_dot: float,  # maximum value of the longitudinal velocity m/s
        min_y_dot_hard_limit: float,  # hard limit value of lateral velocity m/s
        max_y_dot_hard_limit: float,  # hard limit value of lateral velocity m/s
        min_x_dot_dot: float,  # minimum value of the longitudinal acceleration m/s^2
        max_x_dot_dot: float,  # maximum value of the longitudinal acceleration m/s^2
        min_y_dot_dot: float,  # minimum value of the lateral acceleration m/s^2
        max_y_dot_dot: float,  # maximum value of the lateral acceleration m/s^2
        min_d: float,  # minimum steering angle in radians
        max_d: float,  # maximum steering angle in radians
        min_delta_a: float,  # minimum value of the change in acceleration m/s^2
        max_delta_a: float,  # maximum value of the change in acceleration m/s^2
        min_delta_d: float,  # minimum value of the change in steering angle radians/s
        max_delta_d: float,  # maximum value of the change in steering angle radians/s
    ):
        self.min_x_dot = min_x_dot
        self.max_x_dot = max_x_dot
        self.min_y_dot_hard_limit = min_y_dot_hard_limit
        self.max_y_dot_hard_limit = max_y_dot_hard_limit
        self.min_x_dot_dot = min_x_dot_dot
        self.max_x_dot_dot = max_x_dot_dot
        self.min_y_dot_dot = min_y_dot_dot
        self.max_y_dot_dot = max_y_dot_dot
        self.min_d = min_d
        self.max_d = max_d
        self.min_delta_a = min_delta_a
        self.max_delta_a = max_delta_a
        self.min_delta_d = min_delta_d
        self.max_delta_d = max_delta_d

    def attributes_to_ignore(self):
        return []

    def as_dictionary(self):
        return {
            "min_x_dot": self.min_x_dot,
            "max_x_dot": self.max_x_dot,
            "min_y_dot_hard_limit": self.min_y_dot_hard_limit,
            "max_y_dot_hard_limit": self.max_y_dot_hard_limit,
            "min_x_dot_dot": self.min_x_dot_dot,
            "max_x_dot_dot": self.max_x_dot_dot,
            "min_y_dot_dot": self.min_y_dot_dot,
            "max_y_dot_dot": self.max_y_dot_dot,
            "min_d (degrees)": np.degrees(self.min_d),
            "max_d (degrees)": np.degrees(self.max_d),
            "min_delta_a": self.min_delta_a,
            "max_delta_a": self.max_delta_a,
            "min_delta_d (degrees/s)": np.degrees(self.min_delta_d),
            "max_delta_d (degrees/s)": np.degrees(self.max_delta_d),
        }

    def __str__(self):
        return str(self.as_dictionary())

    __repr__ = __str__

    @classmethod
    def from_dict(cls, static_constraints_dict):
        return cls(
            min_x_dot=static_constraints_dict["min_x_dot"],
            max_x_dot=static_constraints_dict["max_x_dot"],
            min_y_dot_hard_limit=static_constraints_dict["min_y_dot_hard_limit"],
            max_y_dot_hard_limit=static_constraints_dict["max_y_dot_hard_limit"],
            min_x_dot_dot=static_constraints_dict["min_x_dot_dot"],
            max_x_dot_dot=static_constraints_dict["max_x_dot_dot"],
            min_y_dot_dot=static_constraints_dict["min_y_dot_dot"],
            max_y_dot_dot=static_constraints_dict["max_y_dot_dot"],
            min_d=static_constraints_dict["min_d"],
            max_d=static_constraints_dict["max_d"],
            min_delta_a=static_constraints_dict["min_delta_a"],
            max_delta_a=static_constraints_dict["max_delta_a"],
            min_delta_d=static_constraints_dict["min_delta_d"],
            max_delta_d=static_constraints_dict["max_delta_d"],
        )