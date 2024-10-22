from copy import deepcopy

import numpy as np
import pandas as pd
from numpy import array


class ControlAction:
    def __init__(self, a: float, d: float):
        """
        Constructor for the ControlAction class.

        @param a: acceleration
        @param d: steering angle (radians)
        """
        self.as_column_vector = array([a, d], dtype=float).reshape(-1, 1)

        self._size = self.as_column_vector.shape[0]

    def as_dictionary(self):
        return {
            "a": self.a,
            "steering_angle (degrees)": np.degrees(self.d),
        }

    def as_series(self):
        return pd.Series(
            {
                "a": self.a,
                "d": self.d,
            }
        )

    @property
    def a(self) -> float:
        return float(self.as_column_vector[0, 0])

    @property
    def d(self) -> float:
        return float(self.as_column_vector[1, 0])

    @property
    def size(self):
        # Count the number of attributes in the ControlAction class
        return self._size

    def copy(self):
        return deepcopy(self)

    def __add__(self, other):
        return ControlAction(a=self.a + other.a, d=self.d + other.d)

    def __mul__(self, other):
        return ControlAction(a=self.a * other, d=self.d * other)

    def __truediv__(self, divisor):
        if isinstance(divisor, (int, float)):
            if divisor == 0.0:
                raise ValueError("Division by zero is not allowed (ControlAction)")
            return ControlAction(a=self.a / divisor, d=self.d / divisor)
        raise ValueError(
            "Division by a non-numeric value is not allowed (ControlAction)"
        )

    def __str__(self):
        return f"U: [a: {self.a}, d: {self.d},].T"