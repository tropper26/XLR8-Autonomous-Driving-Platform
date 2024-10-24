import time
from copy import deepcopy

import numpy as np

from control.base_controller import BaseController
from dto.geometry import Rectangle
from simulation.environment import Environment
from simulation.iteration_info import IterationInfo
from simulation.localisation.kalman_filter import KalmanFilter
from simulation.simulation_result import EndCondition
from simulation.simulation_logger import SimulationLogger
from local_planner.trajectory_planner import TrajectoryPlanner
from state_space.inputs.control_action import ControlAction
from state_space.models.generic_model import GenericModel
from state_space.states.state import State


class Simulation:
    def __init__(
            self,
            controller: BaseController,
            trajectory_planner: TrajectoryPlanner,
            env: Environment,
            sampling_time: float,
            terminate_early: bool = False,
            base_noise_scale: float = 0.0,
            kalman_filter_model: GenericModel = None,
            logger: SimulationLogger = None,
    ):
        self.controller = controller
        self.trajectory_planner = trajectory_planner
        self.env = env
        self.terminate_early = terminate_early
        self.sampling_time = sampling_time
        self.base_noise_scale = base_noise_scale
        self.max_sim_iterations = 100000
        self.logger = logger
        self.iteration_infos = None

        self.kalman_filter = None
        if kalman_filter_model is not None:
            self.kalman_filter = KalmanFilter(model=kalman_filter_model)

    def reset(self):
        self.env.reset()
        self.controller.initialize(
            initial_state=self.env.current_state.copy(),
            starting_control_action=self.env.current_control_action.copy(),
        )
        if self.kalman_filter is not None:
            self.kalman_filter.initialize(
                initial_state=self.env.current_state.copy(),
                starting_control_input=self.env.current_control_action.copy(),
                sampling_time=self.sampling_time,
            )

    def run_sim(self) -> tuple[EndCondition, list[IterationInfo]]:
        self.reset()

        start_time = time.perf_counter()

        iteration_infos: list[IterationInfo] = []

        current_state = self.env.current_state.copy()
        control_action = self.env.current_control_action.copy()
        (
            closest_path_point_index,
            observed_path_discretization,
            observed_path_width,
            visible_obstacles,
        ) = self.env.get_observation()

        self.visible_obstacle_avg: dict[int, (int, Rectangle)] = {}
        for ob in visible_obstacles:
            self.visible_obstacle_avg[ob.id] = (1, ob)

        index = 0
        end_condition = EndCondition.NOT_TERMINATED
        while (
                index < self.max_sim_iterations
                and end_condition == EndCondition.NOT_TERMINATED
        ):
            iteration_start_time = time.time()

            (
                found_clear_trajectory,
                reference_trajectory,
                alternate_trajectories,
                invalid_trajectories,
            ) = self.trajectory_planner.plan_iteration_trajectory(
                current_state=current_state,
                prev_action=control_action,
                observed_path_discretization=observed_path_discretization,
                lane_width_infos=self.env.path.lane_width_infos,
                visible_obstacles=visible_obstacles,
            )

            if reference_trajectory:
                (
                    S_ref,
                    X_ref,
                    Y_ref,
                    Psi_ref,
                    K_ref,
                    x_dot_ref,
                ) = reference_trajectory.discretized.row_at(0, columns=["S", "X", "Y", "Psi", "K", "x_dot"])

                error_state_traj_frame = (
                    self.trajectory_planner.compute_trajectory_frame_error(
                        current_state=current_state,
                        X_traj=X_ref,
                        Y_traj=Y_ref,
                        Psi_traj=Psi_ref,
                        x_dot_traj=x_dot_ref,
                    )
                )

                (
                    control_action,
                    iteration_controller_viz,
                ) = self.controller.compute_action(
                    index,
                    current_state=current_state,
                    error_state=error_state_traj_frame,
                    trajectory_discretization=reference_trajectory.discretized,
                )
            else:
                S_ref, X_ref, Y_ref, Psi_ref, K_ref, x_dot_ref = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

                (
                    control_action,
                    iteration_controller_viz,
                ) = self.controller.stop(current_state, self.sampling_time)
                error_state_traj_frame = State(
                    X=0.0, Y=0.0, Psi=0.0, x_dot=0.0, y_dot=0.0, psi_dot=0.0
                )

                for ob in visible_obstacles:
                    ob.decrement_lifetime(self.sampling_time)

            (
                observable_state,
                observed_control_action,
                next_closest_index_on_path,
                next_observed_path,
                next_observed_path_width,
                noisy_visible_obstacles,
            ) = self.env.step(control_action, base_noise_scale=self.base_noise_scale)

            next_state, next_visible_obstacles = self.correct_observations(
                observable_state, observed_control_action, observed_obstacles=noisy_visible_obstacles,
            )

            end_condition, (reward, reward_explanation) = self.env.check_termination(
                error_state_path_frame=error_state_traj_frame,
                terminate_early=self.terminate_early,
            )

            self.controller.store_transition(
                state=current_state,
                control_action=observed_control_action,
                reward=reward,
                next_state=next_state,
                terminated=end_condition != EndCondition.NOT_TERMINATED,
            )

            S_path, K_path, X_path, Y_path, Psi_path = self.env.path_discretization.row_at(
                closest_path_point_index, columns=("S", "K", "X", "Y", "Psi")
            )

            current_iteration = IterationInfo(
                time=index * self.sampling_time,
                execution_time=time.time() - iteration_start_time,
                X=current_state.X,
                Y=current_state.Y,
                Psi=current_state.Psi,
                x_dot=current_state.x_dot,
                y_dot=current_state.y_dot,
                psi_dot=current_state.psi_dot,
                a=observed_control_action.a,
                d=observed_control_action.d,
                S_ref=S_ref,
                K_ref=K_ref,
                X_ref=X_ref,
                Y_ref=Y_ref,
                Psi_ref=Psi_ref,
                x_dot_ref=x_dot_ref,
                y_dot_ref=0.0,
                psi_dot_ref=0.0,
                error_X=error_state_traj_frame.X,
                error_Y=error_state_traj_frame.Y,
                error_Psi=error_state_traj_frame.Psi,
                error_x_dot=error_state_traj_frame.x_dot,
                error_y_dot=error_state_traj_frame.y_dot,
                error_psi_dot=error_state_traj_frame.psi_dot,
                closest_path_point_index=closest_path_point_index,
                S_path=S_path,
                K_path=K_path,
                X_path=X_path,
                Y_path=Y_path,
                Psi_path=Psi_path,
                reward=reward,
                reward_explanation=reward_explanation,
                controller_viz_info=iteration_controller_viz,
                reference_trajectory=reference_trajectory,
                alternate_trajectories=alternate_trajectories,
                invalid_trajectories=invalid_trajectories,
                visible_obstacles=visible_obstacles,
                found_clear_trajectory=found_clear_trajectory,
            )

            iteration_infos.append(current_iteration)
            if time.perf_counter() - start_time > 3:
                yield end_condition, iteration_infos
                start_time = time.perf_counter()

            current_state = next_state
            closest_path_point_index = next_closest_index_on_path
            observed_path_discretization = next_observed_path
            observed_path_width = next_observed_path_width
            visible_obstacles = next_visible_obstacles
            index += 1

        if index >= self.max_sim_iterations:
            print("Max iterations reached")
        if self.logger is not None:
            self.logger.log_attempt(successful=False)

        yield end_condition, iteration_infos

    def correct_observations(
            self, observed_state: State, observed_control_action: ControlAction, observed_obstacles: list[Rectangle],
    ) -> tuple[State, list[Rectangle]]:
        if self.kalman_filter is None:
            return observed_state, observed_obstacles

        self.kalman_filter.prediction_step(
            observed_longitudinal_a=observed_control_action.a,
            observed_heading_angle=observed_state.Psi,
            observed_steering_angle=observed_control_action.d,
        )

        corrected_state = self.kalman_filter.correction_step(
            observed_position_x=observed_state.X,
            observed_position_y=observed_state.Y,
        )

        for new_ob_measurement in observed_obstacles:
            if new_ob_measurement.id in self.visible_obstacle_avg:
                measurements_count, avg_obst = self.visible_obstacle_avg[new_ob_measurement.id]
                new_x = (measurements_count * avg_obst.x + new_ob_measurement.x) / (measurements_count + 1)
                new_y = (measurements_count * avg_obst.y + new_ob_measurement.y) / (measurements_count + 1)
                new_width = (measurements_count * avg_obst.width + new_ob_measurement.width) / (measurements_count + 1)
                new_height = (measurements_count * avg_obst.height + new_ob_measurement.height) / (
                            measurements_count + 1)

                self.visible_obstacle_avg[new_ob_measurement.id] = (
                    measurements_count + 1, Rectangle(new_x, new_y, new_width, new_height, new_ob_measurement.id, new_ob_measurement.lifetime_seconds))
            else:
                self.visible_obstacle_avg[new_ob_measurement.id] = (1, new_ob_measurement)

        return corrected_state, [ob[1] for ob in self.visible_obstacle_avg.values()]

        # distances_to_obstacles = self.compute_distance_to_obstacles(corrected_state, next_observed_obstacles)

    def compute_distance_to_obstacles(self, current_state: State, obstacles: list[Rectangle]) -> np.ndarray:
        return np.hypot((current_state.X - np.array([ob.x_center for ob in obstacles])),
                        (current_state.Y - np.array([ob.y_center for ob in obstacles])))