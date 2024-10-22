import enum
from copy import deepcopy

import pandas as pd

from control.base_params import BaseControllerParams
from vehicle.vehicle_params import VehicleParams


class SimulationInfo:
    def __init__(
        self,
        sampling_time: float,
        vehicle_model_name: str,
        vehicle_params_name: str,
        static_constraints_name: str,
        controller_name: str,
        controller_params_name: str,
        identifier: str,
        base_noise_scale: float,
        use_kalman_filter: bool,
    ):
        self.sampling_time = sampling_time
        self.vehicle_model_name = vehicle_model_name
        self.vehicle_params_name = vehicle_params_name
        self.static_constraints_name = static_constraints_name
        self.controller_name = controller_name
        self.controller_params_name = controller_params_name
        self.identifier = identifier
        self.base_noise_scale = base_noise_scale
        self.use_kalman_filter = use_kalman_filter

    def get_controller_params(
        self,
        controller_name: str,
        controller_params_name: str,
        current_app_status,
    ) -> BaseControllerParams:
        controller_params_dict = current_app_status.controller_params_lookup[
            controller_name
        ][controller_params_name]

        controller_params_cls: BaseControllerParams = (
            current_app_status.controller_params_cls_lookup[controller_name]
        )
        controller_params: BaseControllerParams = controller_params_cls.from_dict(
            controller_params_dict
        )

        return controller_params

    @classmethod
    def from_dict(
        cls,
        saved_dict: dict,
        identifier: str,
        base_noise_scale: float,
        use_kalman_filter: bool,
    ):
        return cls(
            sampling_time=saved_dict["sampling_time"],
            vehicle_model_name=saved_dict["vehicle_model"],
            vehicle_params_name=saved_dict["vehicle_params"],
            static_constraints_name=saved_dict["static_constraints"],
            controller_name=saved_dict["controller"],
            controller_params_name=saved_dict["controller_params"],
            identifier=identifier,
            base_noise_scale=base_noise_scale,
            use_kalman_filter=use_kalman_filter,
        )

    def validate(self) -> list[str]:
        errors = []
        if self.vehicle_params_name is None:
            errors.append("Vehicle parameters are not set")

        if self.static_constraints_name is None:
            errors.append("Static constraints are not set")

        if self.vehicle_model_name is None:
            errors.append("Vehicle model is not set")

        if self.controller_name is None:
            errors.append("Controller is not set")

        if self.sampling_time is None:
            errors.append("Sampling time is not set")

        return errors

    def __str__(self):
        return f"""
        Sampling Time: {self.sampling_time}
        Vehicle Model: {self.vehicle_model_name}
        Vehicle Params: {self.vehicle_params_name}
        Static Constraints: {self.static_constraints_name}
        Controller: {self.controller_name}
        """

    __repr__ = __str__


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
        vp: VehicleParams,
        iteration_infos: pd.DataFrame,
        ref_df: pd.DataFrame,
        end_condition: EndCondition,
        run_index: int,
    ):
        self.simulation_info = simulation_info
        self.vp = vp
        self.iteration_infos = iteration_infos
        self.ref_df = ref_df
        self.end_condition = end_condition
        self.run_index = run_index

    def __str__(self):
        return f"""
        Simulation Info: {self.simulation_info}
        Vehicle Params: {self.vp}
        Full State DataFrame: {self.iteration_infos}
        Reference DataFrame: {self.ref_df.shape}
        End Condition: {self.end_condition.name.title()}
        """

    def copy(self):
        return deepcopy(self)

    __repr__ = __str__