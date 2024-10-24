from typing import List

import numpy as np
from scipy.interpolate import CubicSpline

from dto.waypoint import Waypoint

def calc_s(x, y):
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.hypot(dx, dy)
    s = [0]
    s.extend(np.cumsum(ds))
    return s

def generate_cubic_spline_trajectory(waypoints: List[Waypoint], sampling_time: float):
    X = [waypoint.x for waypoint in waypoints]
    Y = [waypoint.y for waypoint in waypoints]

    s = list(np.arange(0, calc_s(X, Y)[-1], 0.01))
    X_ref, Y_ref = s, s

    time_array = np.round(np.arange(0, len(X_ref)) * sampling_time, 2)

    return time_array, X_ref, Y_ref


def generate_cubic_spline_v2(waypoints, sampling_time: float):
    # Extract x and y values from waypoints
    x_values = np.array([waypoint.x for waypoint in waypoints])
    y_values = np.array([waypoint.y for waypoint in waypoints])

    # Create a time array corresponding to the waypoints
    t_values = np.arange(len(waypoints))

    # Perform cubic spline interpolation for x and y separately
    spline_x = CubicSpline(t_values, x_values, bc_type="clamped")
    spline_y = CubicSpline(t_values, y_values, bc_type="clamped")

    # Generate a time array with the desired sampling rate
    t_discretized = np.arange(0, len(waypoints) - 1, sampling_time)

    # Evaluate the cubic splines at the discretized time points
    x_discretized = spline_x(t_discretized)
    y_discretized = spline_y(t_discretized)

    return t_discretized, x_discretized, y_discretized