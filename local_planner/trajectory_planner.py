from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from pandas import DataFrame

from dto.coord_transform import inertial_to_path_frame, compute_path_frame_error
from dto.geometry import Rectangle
from dto.waypoint import Waypoint
from local_planner.parametric_curves.spiral_optimisation import (
    create_spiral_interpolation,
    eval_spiral,
)
from local_planner.pathmanager import compute_lateral_points, SpiralInputParams
from state_space.states.state import State
from vehicle.vehicle_info import VehicleInfo


class TrajectoryPlanner:
    def __init__(
        self,
        vi: VehicleInfo,
        min_trajectory_length: float,
        possible_trajectory_step_size: float = 0.5,
    ):
        self.vi = vi
        self.min_trajectory_length = min_trajectory_length
        self.possible_trajectory_step_size = possible_trajectory_step_size

        self.current_trajectory = None
        self.alternate_trajectories = None
        self.invalid_trajectories = None

    def find_closest_index_on_trajectory(self, current_state: State):
        # Compute the Euclidean distance between the vehicle's position and each reference point
        front_X, front_Y = self.vi.front_axle_position(current_state)
        distances = np.linalg.norm(
            np.column_stack((self.current_trajectory.X, self.current_trajectory.Y))
            - np.array([front_X, front_Y]),
            axis=1,
        )

        closest_index = np.argmin(distances)

        return closest_index

    def plan_iteration_trajectory(
        self,
        current_state: State,
        observed_path: DataFrame,
        observed_path_width: float,
        visible_obstacles: list[Rectangle],
    ) -> (DataFrame, bool, list[DataFrame], list[DataFrame]):
        start_S_path = observed_path.iloc[0]["S"]
        end_S_path = observed_path.iloc[-1]["S"]

        if self.current_trajectory is not None:
            closest_index = self.find_closest_index_on_trajectory(current_state)
            self.current_trajectory = self.current_trajectory.iloc[closest_index:]

            trajectory_length = (
                self.current_trajectory.iloc[-1]["S"]
                - self.current_trajectory.iloc[0]["S"]
            )
            path_length = end_S_path - start_S_path

            # If the trajectory is long enough or the remaining path is too short only replan if the path is not clear
            if (
                trajectory_length >= self.min_trajectory_length
                or path_length <= self.min_trajectory_length
            ):
                current_trajectory_clear = not any(
                    np.any(
                        visible_obstacle.intersects_circles(
                            self.current_trajectory[["X", "Y"]].to_numpy(),
                            1.2 * self.vi.vp.width // 2,  # 1.2 is a safety factor
                        )
                    )
                    for visible_obstacle in visible_obstacles
                )
                if current_trajectory_clear:
                    return (
                        self.current_trajectory,
                        True,
                        self.alternate_trajectories,
                        self.invalid_trajectories,
                    )
        print("Replanning trajectory")
        front_X, front_Y = self.vi.front_axle_position(current_state)

        candidate_trajectory_params_list = self.compute_candidate_trajectory_params(
            visible_obstacles,
            observed_path_width,
            observed_path,
            front_X,
            front_Y,
            current_state.Psi,
        )
        (
            best_trajectory_optimisation_params,
            found_clear_trajectory,
            self.alternate_trajectories,
            self.invalid_trajectories,
        ) = self.plan_trajectories_concurrently(
            candidate_trajectory_params_list,
            visible_obstacles,
            step_size=self.possible_trajectory_step_size,
        )

        # compute a more accurate path discretisation than needed for visualisation
        best_trajectory = eval_spiral(
            best_trajectory_optimisation_params,
            x_0=front_X,
            y_0=front_Y,
            psi_0=current_state.Psi,
            ds=0.01,
        )

        best_trajectory["x_dot"] = compute_velocity_profile(
            current_state.x_dot + 0.1,
            v_min=self.vi.min_x_dot,
            v_max=self.vi.max_x_dot,
            a_long_max=self.vi.max_a,
            a_long_min=self.vi.min_a,
            a_lat_max=self.vi.static_constraints.max_y_dot_dot,
            curve=best_trajectory,
        )

        best_trajectory["S"] += start_S_path

        self.current_trajectory = best_trajectory
        return (
            self.current_trajectory,
            found_clear_trajectory,
            self.alternate_trajectories,
            self.invalid_trajectories,
        )

    def compute_candidate_trajectory_params(
        self,
        visible_obstacles,
        observed_path_width,
        observed_path,
        front_X,
        front_Y,
        vehicle_heading,
    ):
        current_arclength = observed_path.iloc[0]["S"]
        target_index = np.searchsorted(
            observed_path.S.values, current_arclength + 5
        )  # get the first index that is ahead of the look ahead distance
        if target_index >= len(observed_path):
            target_index = len(observed_path) - 1
        target_X, target_Y, target_Psi = observed_path.iloc[target_index][
            ["X", "Y", "Psi"]
        ]

        candidate_target_points: list[Waypoint] = [Waypoint(x=target_X, y=target_Y)]

        if visible_obstacles:
            candidate_count = 10
            max_lat_dist = observed_path_width / 2
            distances = np.linspace(0, max_lat_dist, candidate_count // 2)
            distances = np.concatenate((-distances, distances))

            (
                (X_pos, Y_pos),
                (X_neg, Y_neg),
            ) = compute_lateral_points(target_X, target_Y, target_Psi, distances)
            for i in range(len(distances)):
                candidate_target_points.append(Waypoint(x=X_pos[i], y=Y_pos[i]))
                candidate_target_points.append(Waypoint(x=X_neg[i], y=Y_neg[i]))

        candidate_trajectory_params_list = [
            SpiralInputParams(
                x_0=front_X,
                y_0=front_Y,
                psi_0=vehicle_heading,
                k_0=0.0,
                x_f=point.x,
                y_f=point.y,
                psi_f=target_Psi,
                k_f=0.0,
                k_max=1,
            )
            for point in candidate_target_points
        ]
        # print("K_max: ", np.tan(self.vi.max_d) / self.vi.wheelbase)

        return candidate_trajectory_params_list

    def plan_trajectories_concurrently(
        self,
        candidate_trajectory_params_list: list[SpiralInputParams],
        visible_obstacles: list[Rectangle],
        step_size: float = 0.01,
    ):
        best_trajectory_optimisation_params = None
        found_clear_trajectory = False
        alternate_trajectories = []
        invalid_trajectories = []
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self.plan_alternate_trajectory,
                    trajectory_params,
                    visible_obstacles,
                    step_size,
                ): trajectory_params
                for trajectory_params in candidate_trajectory_params_list
            }

            # Iterate over the futures to get the results in order
            p_list = []
            for future in futures:
                trajectory_clear, df, p = future.result()  # wait for the result
                p_list.append(p)
                if trajectory_clear:
                    # possible trajetories are ordered by increasing lateral distance, so first one is the best
                    if best_trajectory_optimisation_params is None:
                        best_trajectory_optimisation_params = p
                    else:
                        alternate_trajectories.append(df)
                else:
                    invalid_trajectories.append(df)
        if best_trajectory_optimisation_params is None:
            print("No clear trajectory found so setting it to p[0]", p_list[0])
            best_trajectory_optimisation_params = p_list[0]
            found_clear_trajectory = False
        return (
            best_trajectory_optimisation_params,
            found_clear_trajectory,
            alternate_trajectories,
            invalid_trajectories,
        )

    def plan_alternate_trajectory(
        self,
        trajectory_params: SpiralInputParams,
        visible_obstacles: list[Rectangle],
        step_size: float,
    ):
        p, df = create_spiral_interpolation(
            *trajectory_params, ds=step_size, equal=False
        )
        circle_centers = df[["X", "Y"]].to_numpy()

        trajectory_clear = not any(
            np.any(
                visible_obstacle.intersects_circles(
                    circle_centers, 1.2 * self.vi.vp.width // 2
                )  # 1.2 is a safety factor
            )
            for visible_obstacle in visible_obstacles
        )

        return trajectory_clear, df, p

    def compute_trajectory_frame_error(
        self, current_state: State, X_traj, Y_traj, Psi_traj, x_dot_traj
    ):
        front_wheel_X, front_wheel_Y = self.vi.front_axle_position(current_state)
        (
            error_X_path_frame,
            error_Y_path_frame,
            error_psi,
            error_x_dot_path_frame,
        ) = compute_path_frame_error(
            X_traj,
            Y_traj,
            Psi_traj,
            front_wheel_X,
            front_wheel_Y,
            current_state.Psi,
            x_dot_traj,
            current_state.x_dot,
        )

        return State(
            error_X_path_frame,
            error_Y_path_frame,
            error_psi,
            error_x_dot_path_frame,
            0.0,
            0.0,
        )


def calc_final_speed(initial_speed, acceleration, distance):
    """
    Computes the new speed and its direction after accelerating or decelerating over a given distance
    based on the initial speed and constant acceleration.

    Parameters
    ----------
    initial_speed : float
        The initial speed in m/s (negative if moving in the reverse direction).
    acceleration : float
        The acceleration in m/s^2 (can be negative if decelerating or moving in the reverse direction).
    distance : float
        The distance over which the acceleration occurs in meters (should be positive).

    Returns
    -------
    float
        The final speed in m/s, positive for forward movement and negative for reverse, based on the inputs.
    """

    squared_term = initial_speed**2 + 2 * acceleration * distance

    # Compute the magnitude of the final speed safely.
    final_speed = np.sqrt(squared_term) if squared_term >= 0 else 0.0

    return final_speed


def compute_velocity_profile(
    v_0, v_min, v_max, a_long_max, a_long_min, a_lat_max, curve: pd.DataFrame
) -> np.ndarray:
    k_max = (
        curve.K.abs().max()
    )  # maximum value of the curvature of the current trajectory
    v_f = np.sqrt(
        a_lat_max / (k_max + 0.1)
    )  # 0.1 is a small value to insane velocity at end of track, where k->0

    v_min = max(min(v_0, v_f), v_min)
    v_max = min(max(v_0, v_f), v_max)

    a_long = a_long_max if (v_f > v_0) else a_long_min  # acceleration/deceleration
    accel_distance = (v_f**2 - v_0**2) / (
        2 * a_long
    )  # distance over which the acceleration/dec occurs

    positions = np.column_stack((curve.X, curve.Y))
    distances = np.linalg.norm(
        np.diff(positions, axis=0), axis=1
    )  # Calculate distances between subsequent points
    cumulative_distances = np.cumsum(
        distances, dtype=float
    )  # Calculate cumulative distances

    velocity_ramp_end_index = np.searchsorted(
        cumulative_distances, accel_distance
    )  # Find index where acceleration distance is reached

    velocity_profile = np.zeros_like(curve.X.values)
    v_i = v_0

    a_long = abs(a_long) if v_f > v_0 else -abs(a_long)

    # Build velocity profile up to ramp end index
    for i in range(velocity_ramp_end_index):
        v_i = calc_final_speed(v_i, a_long, cumulative_distances[i])
        v_i = np.clip(v_i, v_min, v_max)
        velocity_profile[i] = v_i

    # Set velocity profile for the remainder
    velocity_profile[velocity_ramp_end_index:] = v_f

    return velocity_profile