from typing import List

import numpy as np

from dto.waypoint import Waypoint


# def generate_circle_1(waypoints, sampling_time: float):
#     sampling_time = 0.3
#     circle_radius = 1.09
#     circle_arc_length = 1 / 2 * np.pi
#
#     # Calculate the time interval for the circle arc
#     t_circle = np.linspace(0, circle_arc_length, int(circle_arc_length / sampling_time))
#
#     # Generate coordinates for the circle arc
#     x_circle = circle_radius * np.cos(t_circle) - 1.09
#     y_circle = circle_radius * np.sin(t_circle) + 1
#
#     waypoints = [Waypoint(x, y) for x, y in zip(x_circle, y_circle)]
#     return generate_cubic_spline_v2(waypoints=waypoints, sampling_time=sampling_time)
#
#     return t_combined, x_combined, y_combined


def generate_easy_handmade_trajectory(waypoints: List[Waypoint], sampling_time: float):
    duration = 60

    time_array = np.round(
        np.arange(0, int(duration / sampling_time)) * sampling_time, 2
    )

    X = 15 * time_array
    Y = 750 / 900**2 * X**2 + 250

    return time_array, X, Y


def generate_medium_handmade_trajectory(
    waypoints: List[Waypoint], sampling_time: float
):
    duration = 140

    time_array = np.round(
        np.arange(0, int(duration / sampling_time)) * sampling_time, 2
    )

    X1 = 15 * time_array[0 : int(40 / sampling_time + 1)]
    Y1 = (
        50 * np.sin(2 * np.pi * 0.75 / 40 * time_array[0 : int(40 / sampling_time + 1)])
        + 250
    )

    X2 = (
        300
        * np.cos(
            2
            * np.pi
            * 0.5
            / 60
            * (
                time_array[int(40 / sampling_time + 1) : int(100 / sampling_time + 1)]
                - 40
            )
            - np.pi / 2
        )
        + 600
    )
    Y2 = (
        300
        * np.sin(
            2
            * np.pi
            * 0.5
            / 60
            * (
                time_array[int(40 / sampling_time + 1) : int(100 / sampling_time + 1)]
                - 40
            )
            - np.pi / 2
        )
        + 500
    )

    X3 = 600 - 15 * (
        time_array[int(100 / sampling_time + 1) : int(140 / sampling_time + 1)] - 100
    )
    Y3 = (
        50
        * np.cos(
            2
            * np.pi
            * 0.75
            / 40
            * (
                time_array[int(100 / sampling_time + 1) : int(140 / sampling_time + 1)]
                - 100
            )
        )
        + 750
    )

    X = np.concatenate((X1, X2), axis=0)
    Y = np.concatenate((Y1, Y2), axis=0)

    X = np.concatenate((X, X3), axis=0)
    Y = np.concatenate((Y, Y3), axis=0)

    return time_array, X, Y


def generate_hard_handmade_trajectory(waypoints: List[Waypoint], sampling_time: float):
    version = max(len(waypoints), 1)
    duration = 14 * 11

    time_array = np.round(
        np.arange(0, int(duration / sampling_time)) * sampling_time, 2
    )

    f_x = np.array([0, 60, 110, 140, 160, 110, 40, 10, 40, 70, 110, 150]) * version
    f_y = np.array([40, 20, 20, 60, 100, 140, 140, 80, 60, 60, 90, 90]) * version

    # X & Y derivatives
    f_x_dot = np.array([2, 1, 1, 1, 0, -1, -1, 0, 1, 1, 1, 1]) * 3 * version
    f_y_dot = np.array([0, 0, 0, 1, 1, 0, 0, -1, 0, 0, 0, 0]) * 3 * version

    X = []
    Y = []
    sub_trajectory_start_time = np.arange(
        0, 14 * 12, 14
    )  # the first 12 (nr or traj) multiples of 14 ( duration of traj ) are the start times of each map
    for i in range(0, len(sub_trajectory_start_time) - 1):
        # Extract the time elements for each section separately
        if i != len(sub_trajectory_start_time) - 2:
            t_temp = time_array[
                int(sub_trajectory_start_time[i] / sampling_time) : int(
                    sub_trajectory_start_time[i + 1] / sampling_time
                )
            ]
        else:
            t_temp = time_array[
                int(sub_trajectory_start_time[i] / sampling_time) : int(
                    sub_trajectory_start_time[i + 1] / sampling_time + 1
                )
            ]

        # Generate data for a subtrajectory
        M = np.array(
            [
                [1, t_temp[0], t_temp[0] ** 2, t_temp[0] ** 3],
                [1, t_temp[-1], t_temp[-1] ** 2, t_temp[-1] ** 3],
                [0, 1, 2 * t_temp[0], 3 * t_temp[0] ** 2],
                [0, 1, 2 * t_temp[-1], 3 * t_temp[-1] ** 2],
            ]
        )

        c_x = np.array(
            [
                [f_x[i]],
                [f_x[i + 1] - f_x_dot[i + 1] * sampling_time],
                [f_x_dot[i]],
                [f_x_dot[i + 1]],
            ]
        )
        c_y = np.array(
            [
                [f_y[i]],
                [f_y[i + 1] - f_y_dot[i + 1] * sampling_time],
                [f_y_dot[i]],
                [f_y_dot[i + 1]],
            ]
        )

        a_x = np.matmul(np.linalg.inv(M), c_x)
        a_y = np.matmul(np.linalg.inv(M), c_y)

        # Compute X and Y values
        X_temp = (
            a_x[0][0]
            + a_x[1][0] * t_temp
            + a_x[2][0] * t_temp**2
            + a_x[3][0] * t_temp**3
        )
        Y_temp = (
            a_y[0][0]
            + a_y[1][0] * t_temp
            + a_y[2][0] * t_temp**2
            + a_y[3][0] * t_temp**3
        )

        # Concatenate X and Y values
        X = np.concatenate([X, X_temp])
        Y = np.concatenate([Y, Y_temp])

    # Round the numbers to avoid numerical errors
    X = np.round(X, 8)
    Y = np.round(Y, 8)

    return time_array, X, Y