import enum
from copy import deepcopy

from global_planner.path_planning.path import Path
from simulation.simulation_info import SimulationInfo
from vehicle.vehicle_params import VehicleParams
from simulation.iteration_info import IterationInfo, IterationInfoBatch


class EndCondition(enum.Enum):
    DESTINATION_REACHED = 1
    MAX_ITERATIONS_REACHED = 2
    OUT_OF_BOUNDS = 3
    MAX_LATERAL_ERROR = 4
    MAX_HEADING_ERROR = 5
    MAX_VELOCITY_ERROR = 6
    NOT_TERMINATED = 7


class SimulationResult:
    def __init__(
        self,
        simulation_info: SimulationInfo,
        vehicle_params_for_visualization: VehicleParams,
        iteration_infos: list[IterationInfo],
        iteration_info_batch: IterationInfoBatch,
        path: Path,
        end_condition: EndCondition,
        run_index: int,
    ):
        self.simulation_info = simulation_info
        self.vehicle_params_for_visualization = vehicle_params_for_visualization
        self.iteration_infos = iteration_infos
        self.iteration_info_batch = iteration_info_batch
        self.path = path
        self.end_condition = end_condition
        self.run_index = run_index

    def __str__(self):
        return f"""
        Simulation Info: {self.simulation_info}
        Vehicle Params: {self.vehicle_params_for_visualization}
        Iteration Infos: {self.iteration_infos}
        Path Discretization: {len(self.path.discretized)}
        End Condition: {self.end_condition.name.title()}
        """

    def down_sample_for_visualization(self, down_sample_rate: int) -> "SimulationResult":
        last_iteration_info = self.iteration_infos[-1]

        # Downsample the iteration to speed up the animation
        iteration_infos = self.iteration_infos[:: down_sample_rate]

        # Add the last row if it's not already in the down-sampled list
        if iteration_infos[-1].time != last_iteration_info.time:
            iteration_infos.append(last_iteration_info)

        iteration_info_batch = IterationInfoBatch.from_iteration_infos(iteration_infos)

        return SimulationResult(
            simulation_info=self.simulation_info,
            vehicle_params_for_visualization=self.vehicle_params_for_visualization,
            iteration_infos=iteration_infos,
            iteration_info_batch=iteration_info_batch,
            path=self.path,
            end_condition=self.end_condition,
            run_index=self.run_index,
        )

    def copy(self):
        return deepcopy(self)

    __repr__ = __str__