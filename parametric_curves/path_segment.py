from typing import Optional

import numpy as np
from numpy import ndarray

from abc import ABC, abstractmethod

from dto.waypoint import WaypointWithHeading
from parametric_curves.curve import (
    ParametricCurveInfo,
    CurveTypes,
    ParametricCurve,
    CurveDiscretization,
    ParamType,
)


class PathSegmentDiscretization(CurveDiscretization):
    __slots__ = ["lateral_X", "lateral_Y"]

    lateral_X: ndarray
    lateral_Y: ndarray

    def __init__(
        self,
        S: ndarray,
        X: ndarray,
        Y: ndarray,
        Psi: ndarray,
        K: ndarray,
        offset_distances: list[float] = None,
    ):
        super().__init__(S, X, Y, Psi, K)
        if offset_distances is not None:
            self.compute_lateral_points(
                offset_distances,
            )
        else:
            self.lateral_X = None
            self.lateral_Y = None

    def slice(self, start_index: int, amount: int):

        path_disc_slice = PathSegmentDiscretization(
            self.S[start_index : start_index + amount],
            self.X[start_index : start_index + amount],
            self.Y[start_index : start_index + amount],
            self.Psi[start_index : start_index + amount],
            self.K[start_index : start_index + amount],
        )

        if self.lateral_X is not None:
            path_disc_slice.lateral_X = self.lateral_X[start_index : start_index + amount, :]
            path_disc_slice.lateral_Y = self.lateral_Y[start_index : start_index + amount, :]

        return path_disc_slice

    def row_at(self, index: int, columns: tuple = None) -> tuple:
        if columns is None:
            return (
                self.S[index],
                self.X[index],
                self.Y[index],
                self.Psi[index],
                self.K[index],
                self.lateral_X[index, :],
                self.lateral_Y[index, :],
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
                case "lateral_X":
                    result.append(self.lateral_X[index, :])
                case "lateral_Y":
                    result.append(self.lateral_Y[index, :])
                case _:
                    raise ValueError(f"Column '{column}' not recognized.")

        return tuple(result)



    def compute_lateral_points(self, distances: list[float]):
        """
        Compute the lateral positions of a list of points for each distance in the list.
        Args:
            X: the x-coordinates of the points
            Y: the y-coordinates of the points
            Psi: the heading angles of the points
            distances: the distances at which to compute the lateral points

        Returns: two nr_points x nr_distances arrays containing the x and y coordinates of the lateral points
        """
        dX_dt = np.cos(self.Psi)
        dY_dt = np.sin(self.Psi)
        norms = np.sqrt(dX_dt**2 + dY_dt**2)
        nX = -dY_dt / norms
        nY = dX_dt / norms
        # Initialize output arrays
        lateral_X = np.zeros((len(self.X), len(distances)))
        lateral_Y = np.zeros((len(self.Y), len(distances)))

        # Compute lateral points for each distance
        for i, dist in enumerate(distances):
            lateral_X[:, i] = self.X + nX * dist
            lateral_Y[:, i] = self.Y + nY * dist

        self.lateral_X = lateral_X
        self.lateral_Y = lateral_Y


class PathSegmentInfo(ParametricCurveInfo[ParamType]):
    __slots__ = ["offset_distances"]

    offset_distances: list[float]

    def __init__(
        self,
        curve_type: CurveTypes,
        start_point: WaypointWithHeading,
        params: tuple,
        offset_distances: list[float],
    ):
        super().__init__(curve_type, start_point, params)
        self.offset_distances = offset_distances


class PathSegment(ParametricCurve[PathSegmentDiscretization, PathSegmentInfo], ABC):
    def __init__(
        self,
        curve_discretization: Optional[PathSegmentDiscretization] = None,
        curve_info: Optional[PathSegmentInfo] = None,
        discretization_step_size: float = None,
    ):
        super().__init__(curve_info, curve_discretization, discretization_step_size)

    @abstractmethod
    def evaluate(self, eval_step_size: float) -> PathSegmentDiscretization:
        if self.curve_info is None:
            raise ValueError("Cannot evaluate curve without parameters")
        raise NotImplementedError