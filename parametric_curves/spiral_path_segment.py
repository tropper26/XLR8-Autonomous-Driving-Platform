from typing import Optional

import numpy as np

from dto.waypoint import WaypointWithHeading
from parametric_curves.path_segment import (
    CurveTypes,
    PathSegmentDiscretization,
    PathSegment,
    PathSegmentInfo,
)
from rust_switch import eval_spiral

SpiralParams = tuple[float, float, float, float, float]


class SpiralPathSegmentInfo(PathSegmentInfo[SpiralParams]):
    def __init__(
        self,
        start_point: WaypointWithHeading,
        params: SpiralParams,
        offset_distances: list[float],
    ):
        super().__init__(CurveTypes.CUBIC_SPIRAL, start_point, params, offset_distances)


class SpiralPathSegment(PathSegment):
    def __init__(
        self,
        spiral_info: Optional[SpiralPathSegmentInfo] = None,
        curve_discretization: Optional[PathSegmentDiscretization] = None,
    ):
        super().__init__(
            curve_discretization=curve_discretization,
            curve_info=spiral_info,
        )

    def evaluate(self, eval_step_size: float) -> PathSegmentDiscretization:
        if self.curve_info is None:
            raise ValueError("Optimization params not provided")

        s_values, x_values, y_values, psi_values, k_values = eval_spiral(
            p=self.curve_info.params,  # type: ignore
            ds=eval_step_size,
            x_0=self.curve_info.start_point.x,
            y_0=self.curve_info.start_point.y,
            psi_0=self.curve_info.start_point.heading,
        )
        
        s_values = np.asarray(s_values)
        x_values = np.asarray(x_values)
        y_values = np.asarray(y_values)
        psi_values = np.asarray(psi_values)
        k_values = np.asarray(k_values)
        
        self._curve_discretization = PathSegmentDiscretization(
            S=s_values,
            X=x_values,
            Y=y_values,
            Psi=psi_values,
            K=k_values,
        )

        self._curve_discretization.compute_lateral_points(
            self.curve_info.offset_distances
        )

        return self._curve_discretization