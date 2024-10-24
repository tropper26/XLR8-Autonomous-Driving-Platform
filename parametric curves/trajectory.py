from numpy import ndarray

from parametric_curves.curve import CurveDiscretization
from parametric_curves.spiral import ParametricSpiral
from parametric_curves.spiral_optimisation import eval_spiral


class TrajectoryDiscretization(CurveDiscretization):
    __slots__ = ["x_dot"]

    x_dot: ndarray

    def __init__(
            self,
            S: ndarray,
            X: ndarray,
            Y: ndarray,
            Psi: ndarray,
            K: ndarray,
            x_dot: ndarray,
    ):
        super().__init__(S, X, Y, Psi, K)
        self.x_dot = x_dot

    def row_at(self, index: int, columns: list[str] = None) -> tuple:
        if columns is None:
            return (
                self.S[index],
                self.X[index],
                self.Y[index],
                self.Psi[index],
                self.K[index],
                self.x_dot[index],
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
                case "x_dot":
                    result.append(self.x_dot[index])
                case _:
                    raise ValueError(f"Unknown column name: {column}")

        return tuple(result)

    def __getitem__(self, key: slice) -> "TrajectoryDiscretization":
        return TrajectoryDiscretization(
            self.S[key],
            self.X[key],
            self.Y[key],
            self.Psi[key],
            self.K[key],
            self.x_dot[key],
        )

    def slice_inplace(self, key: slice):
        self.S = self.S[key]
        self.X = self.X[key]
        self.Y = self.Y[key]
        self.Psi = self.Psi[key]
        self.K = self.K[key]
        self.x_dot = self.x_dot[key]


class SpiralTrajectory(ParametricSpiral[TrajectoryDiscretization]):
    def evaluate(self, eval_step_size: float) -> TrajectoryDiscretization:
        if self.curve_info is None:
            raise ValueError("Optimization params not provided")

        s_values, x_values, y_values, psi_values, k_values = eval_spiral(
            p=self.curve_info.params,
            ds=eval_step_size,
            x_0=self.curve_info.start_point.x,
            y_0=self.curve_info.start_point.y,
            psi_0=self.curve_info.start_point.heading,
        )

        self._curve_discretization = TrajectoryDiscretization(
            S=s_values,
            X=x_values,
            Y=y_values,
            Psi=psi_values,
            K=k_values,
            x_dot=None,  # Major redesign needed to compute here x_dot instead of assigning it from outside
        )

        return self._curve_discretization