from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from dto.coord_transform import compute_path_frame_error
from dto.geometry import Rectangle
from dto.waypoint import Waypoint, WaypointWithHeading
from global_planner.path_planning import path
from global_planner.path_planning.lane_width_infos import LaneWidthInfos
from parametric_curves.curve import (
    CurveDiscretization,
)
from parametric_curves.path_segment import PathSegmentDiscretization
from parametric_curves.spiral import ParametricSpiralInfo
from parametric_curves.spiral_input_params import SpiralInputParams
from parametric_curves.spiral_optimisation import (
    optimize_spiral,
)
from parametric_curves.trajectory import SpiralTrajectory
from state_space.inputs.control_action import ControlAction
from state_space.states.state import State
from vehicle.vehicle_info import VehicleInfo


def find_closest_index_on_trajectory(
    current_position: Waypoint,
    trajectory: SpiralTrajectory,
):
    # Compute the Euclidean distance between the vehicle's position and each reference point
    # There are other ways to do this, computing the closest point on the parametric
    # representation of the curve instead of the discrete points

    return np.argmin(
        np.hypot(
            trajectory.discretized.X - current_position.x,
            trajectory.discretized.Y - current_position.y,
        )
    )


def compute_lateral_points(X, Y, Psi, dist):
    # Compute the first derivative (tangent vectors)
    dx_dt = np.cos(Psi)
    dy_dt = np.sin(Psi)

    # Calculate normalized normal vectors
    norms = np.hypot(dx_dt, dy_dt)  # Norm of the tangent vectors
    nx = (
        -dy_dt / norms
    ) * dist  # Normal x components (90 degrees rotation of the tangent vector)
    ny = dx_dt / norms * dist

    X = X + nx
    Y = Y + ny

    return X, Y


def compute_candidate_trajectory_params(
    lane_width_infos: LaneWidthInfos,
    observed_path_discretization: PathSegmentDiscretization,
    vehicle_position: WaypointWithHeading,
    look_ahead_distance: float,
    lateral_candidate_count: int,
):
    # get the first index that is ahead of the look ahead distance
    target_index = int(
        observed_path_discretization.S.searchsorted(
            observed_path_discretization.S[0] + look_ahead_distance
        )
    )

    target_index = min(
        target_index, len(observed_path_discretization) - 1
    )  # Clamp target index

    target_X, target_Y, target_Psi = observed_path_discretization.row_at(
        target_index, ("X", "Y", "Psi")
    )

    candidate_target_points: list[Waypoint] = [Waypoint(x=target_X, y=target_Y)]

    for i in range(lateral_candidate_count):
        # get the first index that is ahead of the look ahead distance
        target_index = int(
            observed_path_discretization.S.searchsorted(
                observed_path_discretization.S[0] + look_ahead_distance * ((10 - i - 1) / 10)
            )
        )

        target_index = min(
            target_index, len(observed_path_discretization) - 1
        )  # Clamp target index

        target_X, target_Y, target_Psi = observed_path_discretization.row_at(
            target_index, ("X", "Y", "Psi")
        )

        # candidate_target_points: list[Waypoint] = [Waypoint(x=target_X, y=target_Y)] if i == 0 else []

        lane_centers = path.compute_offset_distances(lane_width_infos.left_lane_count, lane_width_infos.right_lane_count, lane_width_infos.lane_widths)[1::2]

        X, Y = compute_lateral_points(
            target_X, target_Y, target_Psi, np.asarray(lane_centers)
        )

        for i in range(len(lane_centers)):
            candidate_target_points.append(Waypoint(x=X[i], y=Y[i]))

    candidate_trajectory_params_list = [
        SpiralInputParams(
            x_0=vehicle_position.x,
            y_0=vehicle_position.y,
            psi_0=vehicle_position.heading,
            k_0=0.0,
            x_f=point.x,
            y_f=point.y,
            psi_f=target_Psi,
            k_f=0.0,
            k_max=1.0,  # TODO this should actually be: k_max = min{k_max1, k_max2},
            # TODO where k_max1 = tan(max_d)/wheelbase and k_max2 = a_lat_max / (v^2 + 0.0(...)1)
        )
        for point in candidate_target_points
    ]

    return candidate_trajectory_params_list


def optimize_spiral_path_segment(
    segment_params: SpiralInputParams, discretization_step_size: float = None
):
    return SpiralTrajectory(
        spiral_info=ParametricSpiralInfo(
            start_point=WaypointWithHeading(
                x=segment_params.x_0,
                y=segment_params.y_0,
                heading=segment_params.psi_0,
            ),
            params=optimize_spiral(*segment_params.as_tuple()),
        ),
        discretization_step_size=discretization_step_size,
    )


class TrajectoryPlanner:
    def __init__(
        self,
        vi: VehicleInfo,
        min_trajectory_length: float,
        step_size_for_collision_check: float = 0.20,
    ):
        self.vi = vi
        self.min_trajectory_length = min_trajectory_length
        self.step_size_for_collision_check = step_size_for_collision_check
        self.safety_margin_radius = 1.3 * self.vi.vp.width // 2

        self.current_trajectory: Optional[SpiralTrajectory] = None
        self.alternate_trajectories: Optional[list[SpiralTrajectory]] = None
        self.invalid_trajectories: Optional[list[SpiralTrajectory]] = None

    def plan_iteration_trajectory(
        self,
        current_state: State,
        prev_action: ControlAction,
        observed_path_discretization: PathSegmentDiscretization,
        lane_width_infos: LaneWidthInfos,
        visible_obstacles: list[Rectangle],
    ) -> tuple[bool, SpiralTrajectory, list[SpiralTrajectory], list[SpiralTrajectory]]:
        start_S_path = observed_path_discretization.S[0]
        front_X, front_Y = self.vi.front_axle_position(current_state)

        if self.current_trajectory is not None:
            closest_index = find_closest_index_on_trajectory(
                Waypoint(front_X, front_Y), self.current_trajectory
            )

            # remove the points that are behind the vehicle
            self.current_trajectory.discretized.slice_inplace(
                slice(closest_index, None)
            )

            trajectory_length = (
                self.current_trajectory.discretized.S[-1]
                - self.current_trajectory.discretized.S[0]
            )

            path_length = observed_path_discretization.S[-1] - start_S_path

            # If the trajectory is long enough or the remaining path is too short only replan if the path is not clear
            if (
                trajectory_length >= self.min_trajectory_length
                or path_length <= self.min_trajectory_length
            ):
                if check_trajectory_clear(
                    self.current_trajectory.discretized,
                    visible_obstacles,
                    self.safety_margin_radius,
                ):

                    return (
                        True,
                        self.current_trajectory,
                        self.alternate_trajectories,
                        self.invalid_trajectories,
                    )

        print("Replanning trajectory")

        candidate_trajectory_params_list = compute_candidate_trajectory_params(
            lane_width_infos=lane_width_infos,
            observed_path_discretization=observed_path_discretization,
            vehicle_position=WaypointWithHeading(
                front_X, front_Y, current_state.Psi
            ),  # TODO verify if it should be Psi or Psi + steering angle
            look_ahead_distance=5,
            lateral_candidate_count=3 if visible_obstacles else 0,
        )

        (
            found_clear_trajectory,
            best_trajectory,
            self.alternate_trajectories,
            self.invalid_trajectories,
        ) = self.plan_trajectories_concurrently(
            candidate_trajectory_params_list,
            visible_obstacles,
        )

        # compute a more accurate path discretization than needed for visualization
        best_trajectory.evaluate(0.01)

        velocity_profile = compute_velocity_profile(
            current_state.x_dot,
            v_min=self.vi.min_x_dot,
            v_max=self.vi.max_x_dot,
            a_long_max=self.vi.max_a,
            a_long_min=self.vi.min_a,
            a_lat_max=self.vi.static_constraints.max_y_dot_dot,
            trajectory_discretization=best_trajectory.discretized,
        )

        best_trajectory.discretized.x_dot = velocity_profile

        best_trajectory.discretized.S += start_S_path

        self.current_trajectory = best_trajectory

        return (
            found_clear_trajectory,
            self.current_trajectory,
            self.alternate_trajectories,
            self.invalid_trajectories,
        )

    def plan_trajectories_concurrently(
        self,
        candidate_trajectory_params_list: list[SpiralInputParams],
        visible_obstacles: list[Rectangle],
    ) -> (
        bool,
        SpiralTrajectory,
        list[SpiralTrajectory],
        list[SpiralTrajectory],
    ):
        if not candidate_trajectory_params_list:
            raise ValueError("No candidate trajectory parameters provided")

        best_trajectory = None
        found_clear_trajectory = False
        alternate_trajectories: list[SpiralTrajectory] = []
        invalid_trajectories: list[SpiralTrajectory] = []

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self.plan_alternate_trajectory,
                    trajectory_params,
                    visible_obstacles,
                ): trajectory_params
                for trajectory_params in candidate_trajectory_params_list
            }

            # Iterate over the futures to get the results in order
            trajectory_clear: bool
            trajectory: SpiralTrajectory
            for future in futures:
                trajectory_clear, trajectory = future.result()

                if trajectory_clear:
                    # possible trajectories are ordered by increasing lateral distance, so first one is the best
                    if best_trajectory is None:
                        best_trajectory = trajectory
                    else:
                        alternate_trajectories.append(trajectory)
                else:
                    invalid_trajectories.append(trajectory)

        if best_trajectory is None:
            print(
                "No clear trajectory found so setting it to the first invalid trajectory in the list"
            )
            best_trajectory = invalid_trajectories[0]
            found_clear_trajectory = False

        return (
            found_clear_trajectory,
            best_trajectory,
            alternate_trajectories,
            invalid_trajectories,
        )

    def plan_alternate_trajectory(
        self,
        trajectory_params: SpiralInputParams,
        visible_obstacles: list[Rectangle],
    ) -> (bool, SpiralTrajectory):
        trajectory = optimize_spiral_path_segment(
            trajectory_params,
            discretization_step_size=self.step_size_for_collision_check,
        )

        trajectory_clear = check_trajectory_clear(
            trajectory.discretized, visible_obstacles, self.safety_margin_radius
        )

        return trajectory_clear, trajectory

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


def check_trajectory_clear(
    trajectory_discretization: CurveDiscretization,
    visible_obstacles: list[Rectangle],
    margin_radius: float,
):
    circle_centers = np.column_stack(
        (trajectory_discretization.X, trajectory_discretization.Y)
    )

    for obstacle in visible_obstacles:
        if np.any(obstacle.intersects_circles(circle_centers, margin_radius)):
            return False
    return True


def compute_velocity_profile(
    v_0,
    v_min,
    v_max,
    a_long_max,
    a_long_min,
    a_lat_max,
    trajectory_discretization: CurveDiscretization,
) -> np.ndarray:
    k_max = np.max(np.abs(trajectory_discretization.K))

    v_f = np.sqrt(
        a_lat_max / (k_max + 0.1)
    )  # 0.1 is a small value to avoid insane velocity at end of track, where k->0 => v->inf

    v_min = max(min(v_0, v_f), v_min)
    v_max = min(max(v_0, v_f), v_max)

    a_long = a_long_max if (v_f > v_0) else a_long_min  # acceleration/deceleration
    accel_distance = (v_f**2 - v_0**2) / (
        2 * a_long
    )  # distance over which the acceleration/dec occurs

    positions = np.column_stack(
        (trajectory_discretization.X, trajectory_discretization.Y)
    )
    distances = np.linalg.norm(
        np.diff(positions, axis=0), axis=1
    )  # Calculate distances between subsequent points
    cumulative_distances = np.cumsum(
        distances, dtype=float
    )  # Calculate cumulative distances

    velocity_ramp_end_index = np.searchsorted(
        cumulative_distances, accel_distance
    )  # Find index where acceleration distance is reached

    velocity_profile = np.zeros_like(trajectory_discretization.S)
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


def calc_final_speed(
    initial_speed: float, acceleration: float, distance: float | np.ndarray
) -> float | np.ndarray:
    """
    Computes the new speed and its direction after accelerating or decelerating over a given distance
    based on the initial speed and constant acceleration.

    Parameters
    ----------
    initial_speed : float
        The initial speed in m/s (negative if moving in the reverse direction).
    acceleration : float
        The acceleration in m/s^2 (can be negative if decelerating or moving in the reverse direction).
    distance : float | np.ndarray
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