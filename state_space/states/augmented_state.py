from copy import deepcopy

import pandas as pd
from numpy import vstack, array

from state_space.inputs.control_action import ControlAction
from state_space.states.base_state import BaseState
from state_space.states.state import State


class AugmentedState(BaseState):
    def __init__(
        self,
        state: State = None,
        control_inputs: ControlAction = None,
        column_vector_form: array = None,
    ):
        """
        Constructor for the AugmentedState class.

        @param state: the state of the vehicle
        @param control_inputs: the control inputs of the vehicle
        """
        if column_vector_form is not None and (
            state is not None or control_inputs is not None
        ):
            raise ValueError(
                "Cannot specify both column_vector_form and individual state variables."
            )

        if column_vector_form is not None:
            self.as_column_vector = column_vector_form

        else:
            if state is not None and control_inputs is not None:
                self.as_column_vector = vstack(
                    (state.as_column_vector, control_inputs.as_column_vector)  #
                )
            else:
                raise ValueError("Must specify both state and control_inputs.")

        self._size = len(self.as_column_vector)

    @property
    def size(self):
        return self._size

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
    def previous_a(self) -> float:
        return self.as_column_vector[6, 0]

    @property
    def previous_d(self) -> float:
        return self.as_column_vector[7, 0]

    @property
    def prev_U(self):
        return ControlAction(a=self.previous_a, d=self.previous_d)

    def __str__(self):
        return (
            f"X_aug: [X: {self.X:.5f}, Y: {self.Y:.5f}, Psi: {self.Psi:.5f}, x_dot: {self.x_dot:.5f}, y_dot: {self.y_dot:.5f}, "
            f"psi_dot: {self.psi_dot:.5f}, prev_a: {self.previous_a:.5f}, prev_d: {self.previous_d:.5f}]"
        )

    def __add__(self, other):
        return AugmentedState(
            column_vector_form=self.as_column_vector + other.as_column_vector
        )

    def copy(self):
        return deepcopy(self)

    def add_column_vector(self, other: array):
        if other.shape != self.as_column_vector.shape:
            raise ValueError(
                f"Cannot add column vector of shape {other.shape} to state of shape {self.as_column_vector.shape}"
            )
        self.as_column_vector += other

    def as_state(self):
        return State(
            X=self.X,
            Y=self.Y,
            Psi=self.Psi,
            x_dot=self.x_dot,
            y_dot=self.y_dot,
            psi_dot=self.psi_dot,
        )

    def as_series(self) -> pd.Series:
        return pd.Series(
            {
                "X": self.X,
                "Y": self.Y,
                "Psi": self.Psi,
                "x_dot": self.x_dot,
                "y_dot": self.y_dot,
                "psi_dot": self.psi_dot,
                "a": self.previous_a,
                "d": self.previous_d,
            }
        )