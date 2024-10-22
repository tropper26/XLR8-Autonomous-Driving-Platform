from typing import List

import numpy as np
import pandas as pd
from pandas import DataFrame

from dto.waypoint import Waypoint
from local_planner.trajectories.parametric_trajectories import (
    generate_cubic_spline_trajectory,
    generate_cubic_spline_v2,
)
from local_planner.trajectories.simple_trajectories import (
    generate_easy_handmade_trajectory,
    generate_medium_handmade_trajectory,
    generate_hard_handmade_trajectory,
)


def calculate_body_velocities(X, Y, sampling_time):
    X = np.round(X, 8)
    Y = np.round(Y, 8)

    # Calculate the differences in X and Y over time
    dX = np.diff(X)
    dY = np.diff(Y)

    # Calculate X_dot and Y_dot
    X_dot = dX / sampling_time
    Y_dot = dY / sampling_time

    # Pad the first element to maintain the same length
    X_dot = np.concatenate((np.array([X_dot[0]]), X_dot), axis=0)
    Y_dot = np.concatenate((np.array([Y_dot[0]]), Y_dot), axis=0)

    # Calculate second derivatives (accelerations)
    dX_dot = np.diff(X_dot) / sampling_time
    dY_dot = np.diff(Y_dot) / sampling_time

    # Pad the first element to maintain the same length
    dX_dot = np.concatenate((np.array([dX_dot[0]]), dX_dot), axis=0)
    dY_dot = np.concatenate((np.array([dY_dot[0]]), dY_dot), axis=0)

    K = (X_dot * dY_dot - Y_dot * dX_dot) / (X_dot**2 + Y_dot**2 + 0.00001) ** 1.5

    # Calculate reference yaw angles
    Psi = np.arctan2(dY, dX)
    Psi = np.concatenate((np.array([Psi[0]]), Psi), axis=0)

    # We want the yaw angle to keep track the amount of rotations
    PsiIntegrator = Psi
    dPsi = Psi[1 : len(Psi)] - Psi[0 : len(Psi) - 1]
    PsiIntegrator[0] = Psi[0]
    for i in range(1, len(PsiIntegrator)):
        if dPsi[i - 1] < -np.pi:
            PsiIntegrator[i] = PsiIntegrator[i - 1] + (dPsi[i - 1] + 2 * np.pi)
        elif dPsi[i - 1] > np.pi:
            PsiIntegrator[i] = PsiIntegrator[i - 1] + (dPsi[i - 1] - 2 * np.pi)
        else:
            PsiIntegrator[i] = PsiIntegrator[i - 1] + dPsi[i - 1]

    # Calculate body-relative velocities
    ref_x_dot_body = np.cos(PsiIntegrator) * X_dot + np.sin(PsiIntegrator) * Y_dot
    ref_y_dot_body = -np.sin(PsiIntegrator) * X_dot + np.cos(PsiIntegrator) * Y_dot

    time_array = np.arange(0, len(X) * sampling_time, sampling_time)
    time_array = time_array[0 : len(X)]

    df = pd.DataFrame(
        {
            "time": time_array,
            "X": X,
            "Y": Y,
            "Psi": PsiIntegrator,
            "x_dot": ref_x_dot_body,
            "K": K,
        }
    )
    return df


class GlobalTrajectoryPlanner:
    trajectory_strategies = {
        "Cubic Spline ": generate_cubic_spline_trajectory,
        "Cubic Spline V2": generate_cubic_spline_v2,
        "Easy": generate_easy_handmade_trajectory,
        "Medium": generate_medium_handmade_trajectory,
        "Hard": generate_hard_handmade_trajectory,
        "Hard-v2": generate_hard_handmade_trajectory,
    }

    @staticmethod
    def get_possible_strategy_names() -> List[str]:
        return list(GlobalTrajectoryPlanner.trajectory_strategies.keys())

    def __init__(self, sampling_time: float):
        self.sampling_time = sampling_time

    def generate_reference_trajectory(
        self,
        waypoints: List[Waypoint],
        trajectory_strategy_name: str,
    ) -> DataFrame | None:
        # If there are less than 2 waypoints, only hardcoded trajectories are available
        if (not waypoints or len(waypoints) < 2) and trajectory_strategy_name not in [
            "Circle Arc",
            "Easy",
            "Medium",
            "Hard",
            "Hard-v2",
        ]:
            return None

        generate_trajectory_function = GlobalTrajectoryPlanner.trajectory_strategies[
            trajectory_strategy_name
        ]

        if trajectory_strategy_name == "Hard":
            waypoints = [0]  # selecting v1
        elif trajectory_strategy_name == "Hard-v2":
            waypoints = [0, 0, 0]  # selecting v2

        _, X, Y = generate_trajectory_function(waypoints, self.sampling_time)
        df = calculate_body_velocities(X, Y, self.sampling_time)
        return df