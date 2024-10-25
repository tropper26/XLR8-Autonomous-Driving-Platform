import math
from copy import deepcopy

import numpy as np

from dto.coord_transform import path_to_inertial_frame, inertial_to_path_frame
from dto.geometry import Rectangle
from dto.world_bounds import WorldBounds
from dto.waypoint import WaypointWithHeading
from global_planner.path_planning.path import Path
from rust_switch import SpatialGrid
from parametric_curves.path_segment import PathSegmentDiscretization
from simulation.reward_manager import RewardManager
from simulation.simulation_result import EndCondition
from state_space.inputs.control_action import ControlAction
from state_space.models.generic_model import GenericModel
from state_space.state_space import StateSpace
from state_space.states.state import State


class Environment:
    def __init__(
            self,
            reward_manager: RewardManager,
            vehicle_model: GenericModel,
            simulation_x_limit: float,
            simulation_y_limit: float,
            path: Path,
            path_obstacles: list[Rectangle],
            sampling_time: float,
            visible_distance: float,
            grid_cell_count: tuple[int, int] = (32, 32),
            use_random_offset_starting_position=False,
            training=False,
    ):
        self.reward_manager = reward_manager
        self.vehicle_model = vehicle_model
        self.simulation_x_limit = simulation_x_limit
        self.simulation_y_limit = simulation_y_limit
        self.sampling_time = sampling_time
        self.path = path
        self.path_discretization = path.discretized
        self.path_obstacles = path_obstacles
        self.observed_path_width = sum(path.lane_width_infos.lane_widths)
        self.visible_distance = visible_distance

        self.use_random_offset_starting_position = use_random_offset_starting_position
        self.training = training

        self.current_closest_index_on_path = 0
        self.previous_control_action = None

        self.goal_location = WaypointWithHeading(
            x=self.path_discretization.X[-1],
            y=self.path_discretization.Y[-1],
            heading=self.path_discretization.Psi[-1],
        )

        self.setup_grid(grid_cell_count)

        self.initialize_vehicle_position()

    def initialize_vehicle_position(self):
        S, X_path, Y_path, Psi_path, K = self.path_discretization.row_at(
            index=self.current_closest_index_on_path,
            columns=["S", "X", "Y", "Psi", "K"],
        )

        if self.use_random_offset_starting_position:
            Y_vehicle_path_frame = np.random.uniform(-0.8, 0.8)
            Psi_vehicle_path_frame = np.random.uniform(
                -np.radians(-8.6), np.radians(8.6)
            )
            x_dot_vehicle_path_frame = np.random.uniform(-1.0, 1.0)
        else:
            Y_vehicle_path_frame = 0.0
            Psi_vehicle_path_frame = 0.0
            x_dot_vehicle_path_frame = 0.0

        front_x_vehicle, front_y_vehicle, x_dot_vehicle = path_to_inertial_frame(
            X_path,
            Y_path,
            Psi_path,
            0,
            Y_vehicle_path_frame,
            x_dot_vehicle_path_frame,
        )  # in the path frame, the vehicle can only be lateral offset from the path -> x_vehicle_path_frame = 0

        Psi_vehicle = Psi_path - Psi_vehicle_path_frame
        # we want the front axle to be at the path point, so we need to adjust the vehicle position
        x_vehicle = front_x_vehicle - self.vehicle_model.vp.lf * np.cos(Psi_vehicle)
        y_vehicle = front_y_vehicle - self.vehicle_model.vp.lf * np.sin(Psi_vehicle)

        initial_state: State = State(
            X=x_vehicle,
            Y=y_vehicle,
            Psi=Psi_vehicle,
            x_dot=x_dot_vehicle,
            y_dot=0.0,
            psi_dot=0.0,
        )
        starting_control_action = ControlAction(d=0.0, a=0.0)

        self.previous_control_action = starting_control_action

        self.plant = StateSpace(
            model=self.vehicle_model,
            current_state=initial_state.copy(),
            last_input=starting_control_action.copy(),
            sampling_time=self.sampling_time,
        )

    def setup_grid(self, grid_cell_count: tuple[int, int]):
        min_X = 0
        max_X = self.simulation_x_limit
        min_Y = 0
        max_Y = self.simulation_y_limit

        offset_x = (max_X - min_X) / 10
        offset_y = (max_Y - min_Y) / 10

        self.world_bounds = WorldBounds(
            min_X=min_X - offset_x,
            max_X=max_X + offset_x,
            min_Y=min_Y - offset_y,
            max_Y=max_Y + offset_y,
        )

        self.grid = SpatialGrid(
            self.world_bounds.min_X,
            self.world_bounds.max_X,
            self.world_bounds.min_Y,
            self.world_bounds.max_Y,
            grid_cell_count,
        )

        for index_in_path in range(len(self.path_discretization)):
            self.grid.insert_node(
                index_in_path, self.path_discretization.X[index_in_path], self.path_discretization.Y[index_in_path]
            )

    @property
    def current_state(self):
        return self.plant.X

    @property
    def current_control_action(self):
        return self.plant.U

    def _get_closest_index(self, X: float, Y: float) -> int:
        return self.grid.get_closest_node(X, Y)[0]

    def reset(self):
        if (
                not self.training
        ):  # if training, reset the vehicle position to the start of the path
            self.current_closest_index_on_path = 0

        self.initialize_vehicle_position()

    def destination_reached(self, threshold=0.15):
        front_x, front_y = self.plant.model.vp.front_axle_position(self.plant.X)
        x_vehicle, y_vehicle, _ = inertial_to_path_frame(
            x_path=self.goal_location.x,
            y_path=self.goal_location.y,
            psi_path=self.goal_location.heading,
            x_vehicle=front_x,
            y_vehicle=front_y,
            psi_vehicle=self.plant.X.Psi,
        )
        # abs(x_vehicle) < threshold means the front axle is on the line perpendicular to the path at the goal location
        # abs(y_vehicle) < self.observed_path_width / 2 means the front axle is within the path width
        # basically together they check if the car is at the segment that marks the end of the path
        if abs(x_vehicle) < threshold and abs(y_vehicle) < self.observed_path_width / 2:
            return True
        return False

    def get_observation(self) -> (int, Path, float, list[Rectangle]):
        front_wheel_X, front_wheel_Y = self.plant.model.vp.front_axle_position(
            self.plant.X
        )

        current_closest_index_on_path = self._get_closest_index(
            front_wheel_X, front_wheel_Y
        )

        current_length = self.path_discretization.S[current_closest_index_on_path]

        # get the first index that is ahead of the look ahead distance
        target_index = int(self.path_discretization.S.searchsorted(
            current_length + self.visible_distance
        ))

        # Slice the DataFrame between the current closest index and the target index
        observed_path_discretization = self.path_discretization[
                                            current_closest_index_on_path:target_index
                                            ]

        visible_obstacles = []
        for obstacle in self.path_obstacles:
            if obstacle.lifetime_seconds > 0:
                closest_x = np.clip(front_wheel_X, obstacle.x, obstacle.x + obstacle.width)
                closest_y = np.clip(front_wheel_Y, obstacle.y, obstacle.y + obstacle.height)

                # this is for circular visibility
                # if math.hypot(front_wheel_X - closest_x, front_wheel_Y - closest_y) <= self.max_visible_distance:
                #     self.visible_obstacles.append(obstacle)

                # for elliptical visibility

                sin_psi, cos_psi = np.sin(self.plant.X.Psi), np.cos(self.plant.X.Psi)

                closest_x_vehicle_frame = cos_psi * (closest_x - front_wheel_X) + sin_psi * (closest_y - front_wheel_Y)
                closest_y_vehicle_frame = -sin_psi * (closest_x - front_wheel_X) + cos_psi * (closest_y - front_wheel_Y)

                if (closest_x_vehicle_frame ** 2 /
                    self.visible_distance ** 2 + closest_y_vehicle_frame ** 2 / 5 ** 2) <= 1:
                    visible_obstacles.append(deepcopy(obstacle))

        return (
            current_closest_index_on_path,
            observed_path_discretization,
            visible_obstacles,
        )

    def step(
            self, control_action: ControlAction, base_noise_scale=0.0
    ) -> tuple[State, ControlAction, int, PathSegmentDiscretization, float, list[Rectangle]]:
        self.plant.update_control_input(control_action)
        self.plant.propagate_model(self.sampling_time)
        self.current_closest_index_on_path, observed_path_discretization, visible_obstacles = self.get_observation()

        if base_noise_scale == 0.0:
            return (
                self.plant.X,
                self.plant.U,
                self.current_closest_index_on_path,
                observed_path_discretization,
                self.observed_path_width,
                visible_obstacles,
            )

        state_noise, control_noise, obstacle_noises = self._compute_noise(len(visible_obstacles), base_noise_scale)

        distance_to_obstacles = self.compute_distance_to_obstacles(self.plant.X, visible_obstacles)
        distance_scaling_obstacles = self.scale_array(distance_to_obstacles)

        for obstacle_noise, obstacle, distance_scaling in zip(obstacle_noises, visible_obstacles,
                                                                  distance_scaling_obstacles):

            obstacle.x += obstacle_noise.x * distance_scaling
            obstacle.y += obstacle_noise.y * distance_scaling
            obstacle.width += obstacle_noise.width * distance_scaling
            obstacle.height += obstacle_noise.height * distance_scaling

        return (
            self.plant.X + state_noise,
            self.plant.U + control_noise,
            self.current_closest_index_on_path,
            observed_path_discretization,
            self.observed_path_width,
            visible_obstacles,
        )

    @staticmethod
    def scale_array(arr):
        result = np.zeros_like(arr)

        result[arr >= 10] = 1
        result[arr <= 1] = 0.01

        mask = (arr > 1) & (arr < 10)
        result[mask] = 0.01 + (arr[mask] - 1) * (1 - 0.01) / (10 - 1)

        return result

    @staticmethod
    def compute_distance_to_obstacles(current_state: State, obstacles: list[Rectangle]) -> np.ndarray:
        return np.hypot((current_state.X - np.array([ob.x_center for ob in obstacles])),
                        (current_state.Y - np.array([ob.y_center for ob in obstacles])))

    @staticmethod
    def _compute_noise(obstacle_count: int, noise_scale_multiplier=1.0):
        state_noise = State(
            X=np.random.normal(
                0, 1.5 * noise_scale_multiplier
            ),  # GPS position noise (meters)
            Y=np.random.normal(
                0, 1.5 * noise_scale_multiplier
            ),  # GPS position noise (meters)
            Psi=np.random.normal(
                0, np.radians(0.5) * noise_scale_multiplier
            ),  # IMU orientation noise (radians)
            x_dot=np.random.normal(
                0, 0.2 * noise_scale_multiplier
            ),  # Velocity noise (m/s)
            y_dot=np.random.normal(
                0, 0.2 * noise_scale_multiplier
            ),  # Velocity noise (m/s)
            psi_dot=np.random.normal(
                0, np.radians(0.1) * noise_scale_multiplier
            ),  # Gyro noise (radians/s)
        )

        control_noise = ControlAction(
            a=np.random.normal(
                0, 0.1 * noise_scale_multiplier
            ),  # Acceleration noise (m/s²)
            d=np.random.normal(0, np.radians(0.5)),  # Steering angle noise (radians)
        )

        obstacle_noises = []
        for _ in range(obstacle_count):
            obstacle_noise = Rectangle(
                x=np.random.normal(0, 2.50 * noise_scale_multiplier),
                y=np.random.normal(0, 2.50 * noise_scale_multiplier),
                width=np.random.normal(0, 1.25 * noise_scale_multiplier),
                height=np.random.normal(0, 1.25 * noise_scale_multiplier),
            )
            obstacle_noises.append(obstacle_noise)

        return state_noise, control_noise, obstacle_noises

    def check_termination(
            self,
            error_state_path_frame: State,
            terminate_early=False,
    ) -> (EndCondition, (float, str)):
        if error_state_path_frame is None:  # Car is stopped since no trajectory available
            return EndCondition.NOT_TERMINATED, (5, "Car is stopped since no trajectory available")

        if self.destination_reached():
            return (
                EndCondition.DESTINATION_REACHED,
                self.reward_manager.destination_reached_reward(),
            )
        if not self.world_bounds.check_in_bounds(self.plant.X.X, self.plant.X.Y):
            return (
                EndCondition.OUT_OF_BOUNDS,
                self.reward_manager.out_of_bounds_penalty(),
            )

        error_Y_path_frame = error_state_path_frame.Y
        error_psi = error_state_path_frame.Psi
        error_x_dot_path_frame = error_state_path_frame.x_dot

        if terminate_early:
            if (
                    abs(error_Y_path_frame)
                    > self.reward_manager.max_lateral_error_threshold
            ):
                return (
                    EndCondition.MAX_LATERAL_ERROR,
                    self.reward_manager.lateral_error_penalty(error_Y_path_frame),
                )

            if abs(error_psi) > self.reward_manager.max_heading_error_threshold:
                return (
                    EndCondition.MAX_HEADING_ERROR,
                    self.reward_manager.heading_error_penalty(error_psi),
                )

            if (
                    abs(error_x_dot_path_frame)
                    > self.reward_manager.max_velocity_error_threshold
            ):
                return (
                    EndCondition.MAX_VELOCITY_ERROR,
                    self.reward_manager.velocity_error_penalty(error_x_dot_path_frame),
                )

        delta_d = abs(self.previous_control_action.d - self.plant.U.d)
        self.previous_control_action = self.plant.U.copy()

        return EndCondition.NOT_TERMINATED, self.reward_manager.compute_reward(
            error_Y_path_frame, error_psi, error_x_dot_path_frame, delta_d
        )