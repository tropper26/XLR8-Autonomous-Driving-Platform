import json

from control.mpc.mpc import MPC
from control.mpc.mpc_params import MPCParams
from control.pure_pursuit.pure_pursuit import PurePursuitController
from control.pure_pursuit.pure_pursuit_params import PurePursuitParams
from control.sac.sac_controller import SACController
from control.sac.sac_params import SACParams
from control.stanley.stanley import StanleyController
from control.stanley.stanley_params import StanleyParams
from control.td3.td3_controller import TD3Controller
from control.td3.td3_params import TD3Params
from dto.color_theme import ColorTheme
from dto.geometry import Rectangle
from dto.waypoint import WaypointWithHeading
from local_planner.global_trajectory_planner import GlobalTrajectoryPlanner
from global_planner.path_planning.path import Path
from simulation.reward_manager import RewardManager
from simulation.simulation_info import SimulationInfo
from simulation.simulation_result import SimulationResult
from state_space.models.dynamic_bicycle_model import DynamicBicycleModel
from state_space.models.kinematic_bicycle_model import KinematicBicycleModel
from vehicle.static_vehicle_constraints import StaticVehicleConstraints
from vehicle.vehicle_params import VehicleParams


def load_saved_vehicle_params() -> dict:
    vehicle_params_options = {}
    with open(ApplicationStatus.vehicle_params_path, "r") as file:
        vehicle_params_raw = json.load(file)

    for vehicle_name, vehicle_params_dict in vehicle_params_raw.items():
        vehicle_params_options[vehicle_name] = VehicleParams.from_dict(
            vehicle_params_dict
        )

    return vehicle_params_options


def load_saved_static_constraints() -> dict:
    static_constraints_options = {}
    with open(ApplicationStatus.static_constraints_path, "r") as file:
        static_constraints_raw = json.load(file)

    for vehicle_name, static_constraints_dict in static_constraints_raw.items():
        static_constraints_options[vehicle_name] = StaticVehicleConstraints.from_dict(
            static_constraints_dict
        )

    return static_constraints_options


def load_saved_controller_params() -> dict:
    controller_params_options = {}
    with open(ApplicationStatus.controller_params_path, "r") as file:
        controller_params_raw = json.load(file)

    for controller_name, controller_params_dict in controller_params_raw.items():
        controller_params_options[controller_name] = controller_params_dict

    return controller_params_options


def load_global_stylesheet(application_stylesheet_path: str, color_theme: ColorTheme) -> str:
    with open(application_stylesheet_path, 'r') as f:
        stylesheet = f.read()

    global_stylesheet = stylesheet.replace('primary_color', color_theme.primary_color) \
        .replace('secondary_color', color_theme.secondary_color) \
        .replace('button_text_color', color_theme.button_text_color)

    return global_stylesheet


class ApplicationStatus:
    controller_cls_lookup = {
        "Pure Pursuit Controller": PurePursuitController,
        "Stanley Controller": StanleyController,
        "Model Predictive Controller": MPC,
        "Soft Actor Critic": SACController,
        "TD3": TD3Controller,
    }

    controller_options = list(controller_cls_lookup.keys())

    controller_params_cls_lookup = {
        "Pure Pursuit Controller": PurePursuitParams,
        "Stanley Controller": StanleyParams,
        "Model Predictive Controller": MPCParams,
        "Soft Actor Critic": SACParams,
        "TD3": TD3Params,
    }

    vehicle_model_cls_lookup = {
        "Kinematic Bicycle Model": KinematicBicycleModel,
        # "Kinematic Four Wheel Model": GenericModel,
        "Dynamic Bicycle Model": DynamicBicycleModel,
        # "Dynamic Four Wheel Model": GenericModel,
    }

    vehicle_model_options = list(vehicle_model_cls_lookup.keys())

    planning_strategy_options = GlobalTrajectoryPlanner.get_possible_strategy_names()

    vehicle_params_path = "files/vehicle_params.json"
    static_constraints_path = "files/static_constraints.json"
    controller_params_path = "files/controller_params.json"
    application_status_path = "files/application_status.json"
    simulation_info_path = "files/simulation_info.json"
    application_stylesheet_path = "files/style.qss"

    def __init__(self):
        self.color_theme = ColorTheme(
            primary_color="#e02f35",
            secondary_color="#9c9999",
            background_color="#71706e",
            selected_color="#e02f35",
        )
        # self.primary_color = "#fe0002"
        # self.background_color = "#9b9b9b"
        # self.secondary_color = "#c5c5c5"

        self.global_stylesheet = load_global_stylesheet(self.application_stylesheet_path, self.color_theme)
        self.vehicle_params_lookup = load_saved_vehicle_params()
        self.vehicle_params_options = list(self.vehicle_params_lookup.keys())
        self.static_constraints_lookup = load_saved_static_constraints()
        self.static_constraints_options = list(self.static_constraints_lookup.keys())
        self.controller_params_lookup = load_saved_controller_params()

        with open(self.application_status_path, "r") as file:
            application_status_dict = json.load(file)

        self.selected_graph_network_name = application_status_dict["road_network_name"]
        self.planner_sampling_time = application_status_dict["planner_sampling_time"]
        self.planning_strategy_name = application_status_dict["planning_strategy"]

        self.selected_route: list[WaypointWithHeading] = []
        self.initial_world_x_limit = 80
        self.initial_world_y_limit = 45

        self.world_x_limit = self.initial_world_x_limit
        self.world_y_limit = self.initial_world_y_limit

        self.path_obstacles: list[Rectangle] = []
        self.ref_path: Path = None

        self.lane_width = 3.7
        self.lane_count = 2
        self.random_obstacle_count = 0

        self.path_visualisation_step_size = 0.25

        self.min_trajectory_length = 2
        self.possible_trajectory_candidate_count = 20
        self.possible_trajectory_step_size = 0.50

        self.max_visible_distance = 30
        self.reward_manager = RewardManager(
            destination_reached_reward=10,
            out_of_bounds_penalty=-25,
            max_lateral_error_threshold=self.path_width / 2,
            max_lateral_error_penalty=-10,
            max_heading_error_threshold=1.22173,
            max_heading_error_penalty=-10,
            max_velocity_error_threshold=5,
            max_velocity_error_penalty=-10,
        )
        self.environment_grid_cell_count = (64, 64)

        with open(self.simulation_info_path, "r") as file:
            self.simulation_info_dict = json.load(file)

        self.simulation_infos: list[SimulationInfo] = []
        self.simulation_results: dict[str, SimulationResult] = {}

    @property
    def path_width(self):
        return self.lane_count * self.lane_width

    def validate(self) -> list[str]:
        errors = []
        if self.selected_graph_network_name is None:
            errors.append("Graph network is not set")

        if self.planner_sampling_time is None:
            errors.append("Planner sampling time is not set")

        if self.planning_strategy_name is None:
            errors.append("Planning strategy is not set")

        if self.ref_path is None:
            errors.append("Reference trajectory is not set")

        return errors

    @property
    def simulation_count(self):
        return len(self.simulation_infos)

    def add_simulation(self):
        noise_scales = [0.0, 1.0, 1.0]
        use_kalman_filter = [False, True, False]
        print(
            f"Adding simulation {self.simulation_count + 1} with noise scale {noise_scales[self.simulation_count % 3]}"
        )

        new_simulation_info = SimulationInfo.from_dict(
            self.simulation_info_dict,
            identifier=f"Simulation {self.simulation_count + 1}",
            base_noise_scale=noise_scales[self.simulation_count % 3],
            use_kalman_filter=use_kalman_filter[self.simulation_count % 3],
        )
        self.simulation_infos.append(new_simulation_info)
