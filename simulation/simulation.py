import time
import pandas as pd

from control.base_controller import BaseController
from simulation.environment import Environment
from simulation.localisation.kalman_filter import KalmanFilter
from simulation.simulation_info import SimulationResult, EndCondition
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

        data_types = {
            "time": "float64",
            "execution_time": "float64",
            "X": "float64",
            "Y": "float64",
            "Psi": "float64",
            "x_dot": "float64",
            "y_dot": "float64",
            "psi_dot": "float64",
            "a": "float64",
            "d": "float64",
            "S_ref": "float64",
            "K_ref": "float64",
            "X_ref": "float64",
            "Y_ref": "float64",
            "Psi_ref": "float64",
            "x_dot_ref": "float64",
            "y_dot_ref": "float64",
            "psi_dot_ref": "float64",
            "error_X": "float64",
            "error_Y": "float64",
            "error_Psi": "float64",
            "error_x_dot": "float64",
            "error_y_dot": "float64",
            "error_psi_dot": "float64",
            "closest_path_point_index": "int32",
            "S_path": "float64",
            "K_path": "float64",
            "X_path": "float64",
            "Y_path": "float64",
            "Psi_path": "float64",
            "reward": "float64",
            "reward_explaination": "object",
            "controller_viz_info": "object",
            "reference_trajectory": "object",
            "alternate_trajectories": "object",
            "invalid_trajectories": "object",
        }

        self.iteration_infos = pd.DataFrame(
            {
                column: pd.Series(dtype=dtype, index=range(self.max_sim_iterations))
                for column, dtype in data_types.items()
            }
        )

    def run_sim(self) -> SimulationResult:
        self.reset()

        current_state = self.env.current_state.copy()
        (
            closest_path_point_index,
            observed_path,
            observed_path_width,
            visible_obstacles,
        ) = self.env.get_observation()

        index = 0
        end_condition = EndCondition.NOT_TERMINATED

        while (
            index < self.max_sim_iterations
            and end_condition == EndCondition.NOT_TERMINATED
        ):
            iteration_start_time = time.time()

            (
                reference_trajectory,
                found_clear_trajectory,
                alternate_trajectories,
                invalid_trajectories,
            ) = self.trajectory_planner.plan_iteration_trajectory(
                current_state=current_state,
                observed_path=observed_path,
                observed_path_width=observed_path_width,
                visible_obstacles=visible_obstacles,
            )

            (
                S_trajectory,
                X_trajectory,
                Y_trajectory,
                Psi_trajectory,
                K_trajectory,
                x_dot_trajectory,
            ) = reference_trajectory.iloc[0][["S", "X", "Y", "Psi", "K", "x_dot"]]

            error_state_traj_frame = (
                self.trajectory_planner.compute_trajectory_frame_error(
                    current_state=current_state,
                    X_traj=X_trajectory,
                    Y_traj=Y_trajectory,
                    Psi_traj=Psi_trajectory,
                    x_dot_traj=x_dot_trajectory,
                )
            )

            (
                control_action,
                iteration_controller_viz,
            ) = self.controller.compute_action(
                index,
                current_state=current_state,
                error_state=error_state_traj_frame,
                trajectory_df=reference_trajectory,
            )

            (
                observable_state,
                observed_control_action,
                next_closest_index_on_path,
                next_observed_path,
                next_observed_path_width,
                next_visible_obstacles,
            ) = self.env.step(control_action, base_noise_scale=self.base_noise_scale)

            next_state = self.correct_observations(
                observable_state, observed_control_action
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

            S_path, K_path, X_path, Y_path, Psi_path = self.env.full_ref_path.iloc[
                closest_path_point_index
            ][["S", "K", "X", "Y", "Psi"]]

            self.iteration_infos.iloc[index] = pd.Series(
                {
                    "time": index * self.sampling_time,
                    "execution_time": time.time() - iteration_start_time,
                    "X": current_state.X,
                    "Y": current_state.Y,
                    "Psi": current_state.Psi,
                    "x_dot": current_state.x_dot,
                    "y_dot": current_state.y_dot,
                    "psi_dot": current_state.psi_dot,
                    "a": observed_control_action.a,
                    "d": observed_control_action.d,
                    "S_ref": S_trajectory,
                    "K_ref": K_trajectory,
                    "X_ref": X_trajectory,
                    "Y_ref": Y_trajectory,
                    "Psi_ref": Psi_trajectory,
                    "x_dot_ref": x_dot_trajectory,
                    "y_dot_ref": 0.0,
                    "psi_dot_ref": 0.0,
                    "error_X": error_state_traj_frame.X,
                    "error_Y": error_state_traj_frame.Y,
                    "error_Psi": error_state_traj_frame.Psi,
                    "error_x_dot": error_state_traj_frame.x_dot,
                    "error_y_dot": error_state_traj_frame.y_dot,
                    "error_psi_dot": error_state_traj_frame.psi_dot,
                    "closest_path_point_index": closest_path_point_index,
                    "S_path": S_path,
                    "K_path": K_path,
                    "X_path": X_path,
                    "Y_path": Y_path,
                    "Psi_path": Psi_path,
                    "reward": reward,
                    "reward_explaination": reward_explanation,
                    "controller_viz_info": iteration_controller_viz,
                    "reference_trajectory": reference_trajectory,
                    "alternate_trajectories": alternate_trajectories,
                    "invalid_trajectories": invalid_trajectories,
                }
            )

            current_state = next_state
            closest_path_point_index = next_closest_index_on_path
            observed_path = next_observed_path
            observed_path_width = next_observed_path_width
            visible_obstacles = next_visible_obstacles
            index += 1

        if index >= self.max_sim_iterations:
            print("Max iterations reached")
        if self.logger is not None:
            self.logger.log_attempt(successful=False)

        self.iteration_infos = self.iteration_infos[:index]  # remove the extra zeros

        return SimulationResult(
            simulation_info=None,
            vp=None,
            iteration_infos=self.iteration_infos,
            ref_df=self.env.full_ref_path,
            end_condition=end_condition,
            run_index=None,
        )

    def correct_observations(
        self, observed_state: State, observed_control_action: ControlAction
    ) -> State:
        if self.kalman_filter is None:
            return observed_state

        self.kalman_filter.prediction_step(
            observed_longitudinal_a=observed_control_action.a,
            observed_heading_angle=observed_state.Psi,
            observed_steering_angle=observed_control_action.d,
        )

        return self.kalman_filter.correction_step(
            observed_position_x=observed_state.X,
            observed_position_y=observed_state.Y,
        )
