import multiprocessing as mp
import time

from application.application_status import ApplicationStatus
from control.base_controller import BaseController
from control.base_params import BaseControllerParams
from local_planner.trajectory_planner import TrajectoryPlanner
from simulation.environment import Environment
from simulation.simulation import Simulation
from simulation.simulation_info import SimulationInfo
from state_space.models.generic_model import GenericModel
from vehicle.vehicle_info import VehicleInfo
from vehicle.vehicle_params import VehicleParams


class SimulationProcess(mp.Process):
    def __init__(
        self,
        simulation_info: SimulationInfo,
        app_status: ApplicationStatus,
        result_queue: mp.Queue,
        params_pipe: mp.Pipe,
        sim_run_count: mp.Value = mp.Value("i", 1),
        visualization_interval: mp.Value = mp.Value("i", 1),
    ):
        super().__init__()
        self.simulation_info = simulation_info
        self.app_status = app_status
        self.result_queue = result_queue
        self.params_pipe = params_pipe
        self.sim_run_count = sim_run_count
        self.visualization_interval = visualization_interval

        self.daemon = True

        self.running = mp.Value("i", 1)  # Shared value to control the loop
        self.completed_runs = mp.Value("i", 0)  # Shared value to track completed runs

    def run(self):
        try:
            vp, simulation = simulation_setup(self.simulation_info, self.app_status)
            new_params = None
            while self.running.value:
                if self.params_pipe.poll():
                    while self.params_pipe.poll():
                        new_params = self.params_pipe.recv()
                        print("Received new controller parameters: ", new_params)
                    simulation.controller.params = new_params

                if self.completed_runs.value < self.sim_run_count.value:
                    print(
                        f"Running simulation {self.completed_runs.value}:{self.sim_run_count.value}"
                    )
                    sim_result = simulation.run_sim()

                    sim_result.run_index = self.completed_runs.value
                    sim_result.vp = vp
                    sim_result.simulation_info = self.simulation_info
                    if (
                        self.completed_runs.value % self.visualization_interval.value
                        == 0
                    ):
                        self.result_queue.put(sim_result)
                    self.completed_runs.value += 1
                else:
                    time.sleep(0.5)  # Sleep to prevent busy waiting

        except RecursionError as e:
            print("Error in simulation process", e)
            self.result_queue.put({"error": str(e)})

    def terminate(self):
        self.running.value = 0  # Gracefully signal to stop
        super().terminate()  # Force stop if needed


def simulation_setup(
    simulation_info: SimulationInfo, current_app_status: ApplicationStatus
) -> tuple[VehicleParams, Simulation]:
    vp: VehicleParams = current_app_status.vehicle_params_lookup[
        simulation_info.vehicle_params_name
    ]
    static_constraints = current_app_status.static_constraints_lookup[
        simulation_info.static_constraints_name
    ]
    vi = VehicleInfo(vp=vp, constraints=static_constraints)

    model_class = current_app_status.vehicle_model_cls_lookup[
        simulation_info.vehicle_model_name
    ]
    vehicle_model: GenericModel = model_class(vp=vp)

    controller_params: BaseControllerParams = simulation_info.get_controller_params(
        simulation_info.controller_name,
        simulation_info.controller_params_name,
        current_app_status,
    )

    controller_class = current_app_status.controller_cls_lookup[
        simulation_info.controller_name
    ]

    controller: BaseController = controller_class(vi=vi, params=controller_params)

    trajectory_planner = TrajectoryPlanner(
        vi=vi,
        min_trajectory_length=current_app_status.min_trajectory_length,
        possible_trajectory_step_size=current_app_status.possible_trajectory_step_size,
    )

    env = Environment(
        reward_manager=current_app_status.reward_manager,
        vehicle_model=vehicle_model,
        simulation_x_limit=current_app_status.world_x_limit,
        simulation_y_limit=current_app_status.world_y_limit,
        full_ref_path=current_app_status.ref_path_df,
        path_obstacles=current_app_status.path_obstacles,
        path_width=current_app_status.path_width,
        sampling_time=simulation_info.sampling_time,
        grid_cell_count=current_app_status.environment_grid_cell_count,
        max_visible_distance=current_app_status.max_visible_distance,
        use_random_offset_starting_position=False,
        training=False,
    )

    return vp, Simulation(
        controller=controller,
        trajectory_planner=trajectory_planner,
        env=env,
        kalman_filter_model=vehicle_model
        if simulation_info.use_kalman_filter
        else None,
        base_noise_scale=simulation_info.base_noise_scale,
        sampling_time=simulation_info.sampling_time,
        terminate_early=False,
    )