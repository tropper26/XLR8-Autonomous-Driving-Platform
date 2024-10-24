from copy import deepcopy
from enum import Enum
from typing import Optional, TypeVar, Generic

from numpy import ndarray

from abc import ABC, abstractmethod

from dto.waypoint import WaypointWithHeading


class CurveDiscretization:
    __slots__ = [
        "S",
        "X",
        "Y",
        "Psi",
        "K",
    ]

    S: ndarray
    X: ndarray
    Y: ndarray
    Psi: ndarray
    K: ndarray

    def __init__(
        self,
        S: ndarray,
        X: ndarray,
        Y: ndarray,
        Psi: ndarray,
        K: ndarray,
    ):
        self.S = S
        self.X = X
        self.Y = Y
        self.Psi = Psi
        self.K = K

    def __len__(self) -> int:
        return len(self.S)

    def row_at(self, index: int, columns: list[str] = None) -> tuple:
        if columns is None:
            return (
                self.S[index],
                self.X[index],
                self.Y[index],
                self.Psi[index],
                self.K[index],
            )

        result = []

        for column in columns:
            match column:
                case "S":
                    result.append(self.S[index])
                case "X":
                    result.append(self.X[index])
                case "Y":
                    result.append(self.Y[index])
                case "Psi":
                    result.append(self.Psi[index])
                case "K":
                    result.append(self.K[index])
                case _:
                    raise ValueError(f"Column '{column}' not recognized.")

        return tuple(result)

    def __getitem__(self, key: slice) -> "CurveDiscretization":
        return CurveDiscretization(
            self.S[key], self.X[key], self.Y[key], self.Psi[key], self.K[key]
        )

    def slice_inplace(self, key: slice):
        self.S = self.S[key]
        self.X = self.X[key]
        self.Y = self.Y[key]
        self.Psi = self.Psi[key]
        self.K = self.K[key]

    def __str__(self):
        return f"CurveDiscretization(S={self.S}, X={self.X}, Y={self.Y}, Psi={self.Psi}, K={self.K})"

    __repr__ = __str__


class CurveTypes(Enum):
    CUBIC_SPIRAL = 0
    SPLINE = 2
    BEZIER = 3


ParamType = TypeVar("ParamType")


class ParametricCurveInfo(Generic[ParamType]):
    __slots__ = ["curve_type", "start_point", "params"]

    curve_type: CurveTypes
    start_point: WaypointWithHeading
    params: ParamType  # the parameters needed to build the curve from the start_point

    def __init__(
        self,
        curve_type: CurveTypes,
        start_point: WaypointWithHeading,
        params: ParamType,
    ):
        self.curve_type = curve_type
        self.start_point = start_point
        self.params = params

    def __eq__(self, other: "ParametricCurveInfo"):
        return (
            self.curve_type == other.curve_type
            and self.start_point == other.start_point
            and self.params == other.params
        )

    def __str__(self):
        return f"ParametricCurveInfo(curve_type={self.curve_type}, start_point={self.start_point}, params={self.params})"

    __repr__ = __str__


InfoType = TypeVar("InfoType", bound=ParametricCurveInfo)

CurveDisc = TypeVar("CurveDisc", bound=CurveDiscretization)


class ParametricCurve(Generic[CurveDisc, InfoType], ABC):
    __slots__ = [
        "_curve_discretization",
        "curve_params",
        "curve_type",
        "discretization_step_size",
    ]

    _curve_discretization: Optional[CurveDisc]
    curve_info: Optional[InfoType]
    discretization_step_size: float

    def __init__(
        self,
        curve_info: Optional[InfoType] = None,
        curve_discretization: Optional[CurveDisc] = None,
        discretization_step_size: Optional[float] = None,
    ):
        if curve_discretization is not None and curve_info is not None:
            raise ValueError(
                "Only one of parametric_curve or params should be provided"
            )
        elif curve_discretization is None and curve_info is None:
            raise ValueError("One of parametric_curve or params should be provided")

        self.curve_info = curve_info
        self._curve_discretization = curve_discretization

        if discretization_step_size is None:
            self.discretization_step_size = 0.01  # default value
        else:
            self.discretization_step_size = discretization_step_size

    @abstractmethod
    def evaluate(self, eval_step_size: float) -> CurveDisc:
        """
        Compute a discrete representation of the curve based on the optimization parameters and the evaluation step size
        Args:
            eval_step_size: Step size for the arc-length  for the evaluation of the curve [m]

        Returns: A ParametricCurve object of the discrete representation of the curve
        """
        if self.curve_info is None:
            raise ValueError("Cannot evaluate curve without parameters")
        raise NotImplementedError

    @property
    def discretized(self) -> CurveDisc:
        if self._curve_discretization is None:
            self._curve_discretization = self.evaluate(self.discretization_step_size)
        return self._curve_discretization

    def copy(self) -> "ParametricCurve":
        return deepcopy(self)

    def __str__(self):
        return (
            f"params: {self.curve_info}, discretization: \n{self._curve_discretization}"
        )

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other: "ParametricCurve"):
        return self.curve_info == other.curve_info