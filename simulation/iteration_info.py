from copy import deepcopy

from control.controller_viz_info import ControllerVizInfo
from dto.geometry import Rectangle
from parametric_curves.spiral import ParametricSpiral
import numpy as np


class IterationInfo:
    __slots__ = [
        "time",
        "execution_time",
        "X",
        "Y",
        "Psi",
        "x_dot",
        "y_dot",
        "psi_dot",
        "a",
        "d",
        "S_ref",
        "K_ref",
        "X_ref",
        "Y_ref",
        "Psi_ref",
        "x_dot_ref",
        "y_dot_ref",
        "psi_dot_ref",
        "error_X",
        "error_Y",
        "error_Psi",
        "error_x_dot",
        "error_y_dot",
        "error_psi_dot",
        "closest_path_point_index",
        "S_path",
        "K_path",
        "X_path",
        "Y_path",
        "Psi_path",
        "reward",
        "reward_explanation",
        "controller_viz_info",
        "reference_trajectory",
        "alternate_trajectories",
        "invalid_trajectories",
        "visible_obstacles",
        "found_clear_trajectory",
    ]

    time: float
    execution_time: float
    X: float
    Y: float
    Psi: float
    x_dot: float
    y_dot: float
    psi_dot: float
    a: float
    d: float
    S_ref: float
    K_ref: float
    X_ref: float
    Y_ref: float
    Psi_ref: float
    x_dot_ref: float
    y_dot_ref: float
    psi_dot_ref: float
    error_X: float
    error_Y: float
    error_Psi: float
    error_x_dot: float
    error_y_dot: float
    error_psi_dot: float
    closest_path_point_index: int
    S_path: float
    K_path: float
    X_path: float
    Y_path: float
    Psi_path: float
    reward: float
    reward_explanation: str
    controller_viz_info: ControllerVizInfo
    reference_trajectory: ParametricSpiral
    alternate_trajectories: list[ParametricSpiral]
    invalid_trajectories: list[ParametricSpiral]
    visible_obstacles: list[Rectangle]
    found_clear_trajectory: bool

    def __init__(
            self,
            time: float = None,
            execution_time: float = None,
            X: float = None,
            Y: float = None,
            Psi: float = None,
            x_dot: float = None,
            y_dot: float = None,
            psi_dot: float = None,
            a: float = None,
            d: float = None,
            S_ref: float = None,
            K_ref: float = None,
            X_ref: float = None,
            Y_ref: float = None,
            Psi_ref: float = None,
            x_dot_ref: float = None,
            y_dot_ref: float = None,
            psi_dot_ref: float = None,
            error_X: float = None,
            error_Y: float = None,
            error_Psi: float = None,
            error_x_dot: float = None,
            error_y_dot: float = None,
            error_psi_dot: float = None,
            closest_path_point_index: int = None,
            S_path: float = None,
            K_path: float = None,
            X_path: float = None,
            Y_path: float = None,
            Psi_path: float = None,
            reward: float = None,
            reward_explanation: str = None,
            controller_viz_info: ControllerVizInfo = None,
            reference_trajectory: ParametricSpiral = None,
            alternate_trajectories: list[ParametricSpiral] = None,
            invalid_trajectories: list[ParametricSpiral] = None,
            visible_obstacles: list[Rectangle] = None,
            found_clear_trajectory: bool = None,

    ):
        self.time = time
        self.execution_time = execution_time
        self.X = X
        self.Y = Y
        self.Psi = Psi
        self.x_dot = x_dot
        self.y_dot = y_dot
        self.psi_dot = psi_dot
        self.a = a
        self.d = d
        self.S_ref = S_ref
        self.K_ref = K_ref
        self.X_ref = X_ref
        self.Y_ref = Y_ref
        self.Psi_ref = Psi_ref
        self.x_dot_ref = x_dot_ref
        self.y_dot_ref = y_dot_ref
        self.psi_dot_ref = psi_dot_ref
        self.error_X = error_X
        self.error_Y = error_Y
        self.error_Psi = error_Psi
        self.error_x_dot = error_x_dot
        self.error_y_dot = error_y_dot
        self.error_psi_dot = error_psi_dot
        self.closest_path_point_index = closest_path_point_index
        self.S_path = S_path
        self.K_path = K_path
        self.X_path = X_path
        self.Y_path = Y_path
        self.Psi_path = Psi_path
        self.reward = reward
        self.reward_explanation = reward_explanation
        self.controller_viz_info = controller_viz_info
        self.reference_trajectory = reference_trajectory
        self.alternate_trajectories = alternate_trajectories
        self.invalid_trajectories = invalid_trajectories
        self.visible_obstacles = visible_obstacles
        self.found_clear_trajectory = found_clear_trajectory

    def as_table(self, columns_to_display: list[str]) -> list:
        values_for_columns = []
        for column in columns_to_display:
            values_for_columns.append(getattr(self, column))

        return values_for_columns


class IterationInfoBatch:
    __slots__ = [
        "time",
        "execution_time",
        "X",
        "Y",
        "Psi",
        "x_dot",
        "y_dot",
        "psi_dot",
        "a",
        "d",
        "S_ref",
        "K_ref",
        "X_ref",
        "Y_ref",
        "Psi_ref",
        "x_dot_ref",
        "y_dot_ref",
        "psi_dot_ref",
        "error_X",
        "error_Y",
        "error_Psi",
        "error_x_dot",
        "error_y_dot",
        "error_psi_dot",
        "closest_path_point_index",
        "S_path",
        "K_path",
        "X_path",
        "Y_path",
        "Psi_path",
        "reward",
        "reward_explanation",
        "controller_viz_info",
        "reference_trajectory",
        "alternate_trajectories",
        "invalid_trajectories",
        "visible_obstacles",
        "found_clear_trajectory",
    ]

    time: np.ndarray
    execution_time: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    Psi: np.ndarray
    x_dot: np.ndarray
    y_dot: np.ndarray
    psi_dot: np.ndarray
    a: np.ndarray
    d: np.ndarray
    S_ref: np.ndarray
    K_ref: np.ndarray
    X_ref: np.ndarray
    Y_ref: np.ndarray
    Psi_ref: np.ndarray
    x_dot_ref: np.ndarray
    y_dot_ref: np.ndarray
    psi_dot_ref: np.ndarray
    error_X: np.ndarray
    error_Y: np.ndarray
    error_Psi: np.ndarray
    error_x_dot: np.ndarray
    error_y_dot: np.ndarray
    error_psi_dot: np.ndarray
    closest_path_point_index: np.ndarray
    S_path: np.ndarray
    K_path: np.ndarray
    X_path: np.ndarray
    Y_path: np.ndarray
    Psi_path: np.ndarray
    reward: np.ndarray
    reward_explanation: np.ndarray
    controller_viz_info: np.ndarray
    reference_trajectory: np.ndarray
    alternate_trajectories: np.ndarray
    invalid_trajectories: np.ndarray
    visible_obstacles: np.ndarray
    found_clear_trajectory: np.ndarray

    def __init__(self, n: int):
        """
        Initializes an empty IterationInfoBatch with pre-allocated arrays of size `n`.
        """
        self.time = np.empty(n)
        self.execution_time = np.empty(n)
        self.X = np.empty(n)
        self.Y = np.empty(n)
        self.Psi = np.empty(n)
        self.x_dot = np.empty(n)
        self.y_dot = np.empty(n)
        self.psi_dot = np.empty(n)
        self.a = np.empty(n)
        self.d = np.empty(n)
        self.S_ref = np.empty(n)
        self.K_ref = np.empty(n)
        self.X_ref = np.empty(n)
        self.Y_ref = np.empty(n)
        self.Psi_ref = np.empty(n)
        self.x_dot_ref = np.empty(n)
        self.y_dot_ref = np.empty(n)
        self.psi_dot_ref = np.empty(n)
        self.error_X = np.empty(n)
        self.error_Y = np.empty(n)
        self.error_Psi = np.empty(n)
        self.error_x_dot = np.empty(n)
        self.error_y_dot = np.empty(n)
        self.error_psi_dot = np.empty(n)
        self.closest_path_point_index = np.empty(n, dtype=int)
        self.S_path = np.empty(n)
        self.K_path = np.empty(n)
        self.X_path = np.empty(n)
        self.Y_path = np.empty(n)
        self.Psi_path = np.empty(n)
        self.reward = np.empty(n)
        self.reward_explanation = np.empty(n, dtype=str)
        self.controller_viz_info = np.empty(n, dtype=ControllerVizInfo)
        self.reference_trajectory = np.empty(n, dtype=ParametricSpiral)
        self.alternate_trajectories = np.empty(n, dtype=list)
        self.invalid_trajectories = np.empty(n, dtype=list)
        self.visible_obstacles = np.empty(n, dtype=list)
        self.found_clear_trajectory = np.empty(n, dtype=bool)

    @classmethod
    def from_iteration_infos(cls, iteration_infos: list[IterationInfo]) -> "IterationInfoBatch":
        """
        Class method to create an IterationInfoBatch instance from a list of IterationInfo objects.
        This method avoids element-by-element assignment by leveraging Numpy's array operations.
        """
        n = len(iteration_infos)
        batch = cls(n)  # Create an empty batch with size n

        # Use list comprehensions to extract values in bulk
        batch.time[:] = [info.time for info in iteration_infos]
        batch.execution_time[:] = [info.execution_time for info in iteration_infos]
        batch.X[:] = [info.X for info in iteration_infos]
        batch.Y[:] = [info.Y for info in iteration_infos]
        batch.Psi[:] = [info.Psi for info in iteration_infos]
        batch.x_dot[:] = [info.x_dot for info in iteration_infos]
        batch.y_dot[:] = [info.y_dot for info in iteration_infos]
        batch.psi_dot[:] = [info.psi_dot for info in iteration_infos]
        batch.a[:] = [info.a for info in iteration_infos]
        batch.d[:] = [info.d for info in iteration_infos]
        batch.S_ref[:] = [info.S_ref for info in iteration_infos]
        batch.K_ref[:] = [info.K_ref for info in iteration_infos]
        batch.X_ref[:] = [info.X_ref for info in iteration_infos]
        batch.Y_ref[:] = [info.Y_ref for info in iteration_infos]
        batch.Psi_ref[:] = [info.Psi_ref for info in iteration_infos]
        batch.x_dot_ref[:] = [info.x_dot_ref for info in iteration_infos]
        batch.y_dot_ref[:] = [info.y_dot_ref for info in iteration_infos]
        batch.psi_dot_ref[:] = [info.psi_dot_ref for info in iteration_infos]
        batch.error_X[:] = [info.error_X for info in iteration_infos]
        batch.error_Y[:] = [info.error_Y for info in iteration_infos]
        batch.error_Psi[:] = [info.error_Psi for info in iteration_infos]
        batch.error_x_dot[:] = [info.error_x_dot for info in iteration_infos]
        batch.error_y_dot[:] = [info.error_y_dot for info in iteration_infos]
        batch.error_psi_dot[:] = [info.error_psi_dot for info in iteration_infos]
        batch.closest_path_point_index[:] = [info.closest_path_point_index for info in iteration_infos]
        batch.S_path[:] = [info.S_path for info in iteration_infos]
        batch.K_path[:] = [info.K_path for info in iteration_infos]
        batch.X_path[:] = [info.X_path for info in iteration_infos]
        batch.Y_path[:] = [info.Y_path for info in iteration_infos]
        batch.Psi_path[:] = [info.Psi_path for info in iteration_infos]
        batch.reward[:] = [info.reward for info in iteration_infos]
        batch.reward_explanation[:] = [info.reward_explanation for info in iteration_infos]
        batch.controller_viz_info[:] = [info.controller_viz_info for info in iteration_infos]
        batch.reference_trajectory[:] = [info.reference_trajectory for info in iteration_infos]
        batch.alternate_trajectories[:] = [info.alternate_trajectories for info in iteration_infos]
        batch.invalid_trajectories[:] = [info.invalid_trajectories for info in iteration_infos]
        batch.visible_obstacles[:] = [info.visible_obstacles for info in iteration_infos]
        batch.found_clear_trajectory[:] = [info.found_clear_trajectory for info in iteration_infos]

        return batch

    def __getitem__(self, slice_key: slice) -> "IterationInfoBatch":
        sliced_batch = IterationInfoBatch(self.time[slice_key].shape[0])

        sliced_batch.time = self.time[slice_key]
        sliced_batch.execution_time = self.execution_time[slice_key]
        sliced_batch.X = self.X[slice_key]
        sliced_batch.Y = self.Y[slice_key]
        sliced_batch.Psi = self.Psi[slice_key]
        sliced_batch.x_dot = self.x_dot[slice_key]
        sliced_batch.y_dot = self.y_dot[slice_key]
        sliced_batch.psi_dot = self.psi_dot[slice_key]
        sliced_batch.a = self.a[slice_key]
        sliced_batch.d = self.d[slice_key]
        sliced_batch.S_ref = self.S_ref[slice_key]
        sliced_batch.K_ref = self.K_ref[slice_key]
        sliced_batch.X_ref = self.X_ref[slice_key]
        sliced_batch.Y_ref = self.Y_ref[slice_key]
        sliced_batch.Psi_ref = self.Psi_ref[slice_key]
        sliced_batch.x_dot_ref = self.x_dot_ref[slice_key]
        sliced_batch.y_dot_ref = self.y_dot_ref[slice_key]
        sliced_batch.psi_dot_ref = self.psi_dot_ref[slice_key]
        sliced_batch.error_X = self.error_X[slice_key]
        sliced_batch.error_Y = self.error_Y[slice_key]
        sliced_batch.error_Psi = self.error_Psi[slice_key]
        sliced_batch.error_x_dot = self.error_x_dot[slice_key]
        sliced_batch.error_y_dot = self.error_y_dot[slice_key]
        sliced_batch.error_psi_dot = self.error_psi_dot[slice_key]
        sliced_batch.closest_path_point_index = self.closest_path_point_index[slice_key]
        sliced_batch.S_path = self.S_path[slice_key]
        sliced_batch.K_path = self.K_path[slice_key]
        sliced_batch.X_path = self.X_path[slice_key]
        sliced_batch.Y_path = self.Y_path[slice_key]
        sliced_batch.Psi_path = self.Psi_path[slice_key]
        sliced_batch.reward = self.reward[slice_key]
        sliced_batch.reward_explanation = self.reward_explanation[slice_key]
        sliced_batch.controller_viz_info = self.controller_viz_info[slice_key]
        sliced_batch.reference_trajectory = self.reference_trajectory[slice_key]
        sliced_batch.alternate_trajectories = self.alternate_trajectories[slice_key]
        sliced_batch.invalid_trajectories = self.invalid_trajectories[slice_key]
        sliced_batch.visible_obstacles = self.visible_obstacles[slice_key]
        sliced_batch.found_clear_trajectory = self.found_clear_trajectory[slice_key]

        return sliced_batch