from typing import Optional

from dto.waypoint import WaypointWithHeading
from parametric_curves.curve import (
    CurveDiscretization,
    ParametricCurve,
    CurveTypes,
    ParametricCurveInfo, CurveDisc,
)
from parametric_curves.spiral_optimisation import eval_spiral

SpiralParams = tuple[float, float, float, float, float]


class ParametricSpiralInfo(ParametricCurveInfo[SpiralParams]):
    def __init__(
        self,
        start_point: WaypointWithHeading,
        params: SpiralParams,
    ):
        super().__init__(
            CurveTypes.CUBIC_SPIRAL,
            start_point,
            params,
        )


class ParametricSpiral(ParametricCurve[CurveDisc, ParametricSpiralInfo]):
    def __init__(
        self,
        curve_discretization: Optional[CurveDiscretization] = None,
        spiral_info: Optional[ParametricSpiralInfo] = None,
        discretization_step_size: Optional[float] = None,
    ):
        super().__init__(
            curve_info=spiral_info,
            curve_discretization=curve_discretization,
            discretization_step_size=discretization_step_size,
        )

    def evaluate(self, eval_step_size: float) -> CurveDiscretization:
        if self.curve_info is None:
            raise ValueError("Optimization params not provided")

        s_values, x_values, y_values, psi_values, k_values = eval_spiral(
            p=self.curve_info.params,  # type: ignore
            ds=eval_step_size,
            x_0=self.curve_info.start_point.x,
            y_0=self.curve_info.start_point.y,
            psi_0=self.curve_info.start_point.heading,
        )

        self._curve_discretization = CurveDiscretization(
            S=s_values,
            X=x_values,
            Y=y_values,
            Psi=psi_values,
            K=k_values,
        )

        return self._curve_discretization

    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return super().__repr__()