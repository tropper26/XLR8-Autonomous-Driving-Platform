import json
import time

from control.mpc.mpc_params import MPCParams
from control.mpc.weights import Weights
from vehicle.vehicle_info import VehicleInfo
from vehicle.vehicle_params import VehicleParams


class SimulationLogger:
    def __init__(
        self,
        sampling_time: float,
        weights: Weights,
        vehicle_info: VehicleParams,
        constraints: VehicleInfo,
        params: MPCParams,
        trajectory_name: str,
        version: int,
        visualize: bool = False,
    ):
        self.time = time.time()
        self.sampling_time = sampling_time
        self.weights = weights
        self.vehicle_info = vehicle_info
        self.constraints = constraints
        self.params = params
        self.trajectory_name = trajectory_name
        self.version = version
        self.visualize = visualize

    def log_attempt(self, successful: bool, error: str = ""):
        """Serialize the current instance to JSON and append it to a file."""
        human_readable_time = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(self.time)
        )

        data = {
            "time": human_readable_time,
            "trajectory_name": self.trajectory_name,
            "version": self.version,
            "successful": successful,
            "error": error,
            "sampling_time": self.sampling_time,
            "weights": self.weights.__dict__,  # Assuming Weights has a to_dict() method
            "vehicle_info": self.vehicle_info.as_dictionary(),  # Assuming VehicleParams has a to_dict() method
            "constraints": self.constraints.as_dictionary(),  # Assuming VehicleInfo has a to_dict() method
            "params": self.params.__dict__,  # Assuming MPCParams has a to_dict() method
            "visualize": self.visualize,
        }

        log_file = "files/simulation_log.json"
        try:
            with open(log_file, "r") as file:
                log_data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            log_data = []

        log_data.append(data)

        with open(log_file, "w") as file:
            json.dump(log_data, file, indent=4)