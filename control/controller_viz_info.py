from enum import Enum, auto

import numpy as np


class Types(Enum):
    Point = auto()
    Line = auto()


class ControllerVizInfo:
    def __init__(
        self,
        viz_type: Types,
        X: np.ndarray,
        Y: np.ndarray,
        ref_X: np.ndarray = None,
        ref_Y: np.ndarray = None,
    ):
        self.viz_type = viz_type
        self.X = X
        self.Y = Y
        self.ref_X = ref_X
        self.ref_Y = ref_Y

    def __repr__(self):
        return f"ControllerVizInfo({self.X}, {self.Y}, {self.viz_type})"

    __str__ = __repr__