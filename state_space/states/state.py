from copy import deepcopy

import pandas as pd
from numpy import array

from state_space.states.base_state import BaseState


class State(BaseState):
    def __init__(
        self,
        X: float = None,
        Y: float = None,
        Psi: float = None,
        x_dot: float = None,
        y_dot: float = None,
        psi_dot: float = None,
        column_vector_form: array = None,
    ):
        """
        | Constructor for the State class.
        |
        | :param x_dot: longitudinal velocity in the body frame
        | :param y_dot: lateral velocity in the body frame
        | :param Psi:  heading angle
        | :param psi_dot: angular velocity
        | :param X: longitudinal position in the global/inertial frame
        | :param Y: lateral position in the global/inertial frame
        """
        if column_vector_form is not None and (
            X is not None
            or Y is not None
            or Psi is not None
            or x_dot is not None
            or y_dot is not None
            or psi_dot is not None
        ):
            raise ValueError(
                "Cannot specify both column_vector_form and individual state variables."
            )
        if column_vector_form is not None:
            self.as_column_vector = column_vector_form
        else:
            self.as_column_vector = array([X, Y, Psi, x_dot, y_dot, psi_dot]).reshape(
                -1, 1
            )
        self._size = len(self.as_column_vector)

    @property
    def X(self):
        return self.as_column_vector[0, 0]

    @property
    def Y(self):
        return self.as_column_vector[1, 0]

    @property
    def Psi(self):
        return self.as_column_vector[2, 0]

    @property
    def x_dot(self):
        return self.as_column_vector[3, 0]

    @property
    def y_dot(self):
        return self.as_column_vector[4, 0]

    @property
    def psi_dot(self):
        return self.as_column_vector[5, 0]

    @property
    def size(self):
        return self._size

    def __str__(self):
        return f"X: [X: {self.X:.5f}, Y: {self.Y:.5f}, Psi: {self.Psi:.5f}, x_dot: {self.x_dot:.5f}, y_dot: {self.y_dot:.5f}, psi_dot: {self.psi_dot:.5f}]"

    __repr__ = __str__

    def __add__(self, other):
        return State(column_vector_form=self.as_column_vector + other.as_column_vector)

    def copy(self):
        return deepcopy(self)

    def add_column_vector(self, other: array):
        if other.shape != self.as_column_vector.shape:
            raise ValueError(
                f"Cannot add column vector of shape {other.shape} to state of shape {self.as_column_vector.shape}"
            )
        self.as_column_vector += other

    def as_series(self) -> pd.Series:
        """
        Convert the state into a pandas Series object.
        """
        return pd.Series(
            {
                "X": self.X,
                "Y": self.Y,
                "Psi": self.Psi,
                "x_dot": self.x_dot,
                "y_dot": self.y_dot,
                "psi_dot": self.psi_dot,
            }
        )