import numpy as np


class Vector3:
    def __init__(self, data):
        if len(data) != 3:
            raise ValueError(
                "Vector3 must be initialized with a sequence of three elements."
            )
        self.as_column_vector = np.array([[data[0]], [data[1]], [data[2]]])

    @property
    def x(self):
        return self.as_column_vector[0][0]

    @property
    def y(self):
        return self.as_column_vector[1][0]

    @property
    def z(self):
        return self.as_column_vector[2][0]

    def __str__(self):
        return f"Vector3({self.x}, {self.y}, {self.z})"

    def __repr__(self):
        return str(self)