from abc import ABC, abstractmethod

from control.base_params import BaseControllerParams
from control.controller_viz_info import ControllerVizInfo
from parametric_curves.trajectory import TrajectoryDiscretization
from state_space.inputs.control_action import ControlAction
from state_space.states.state import State
from vehicle.vehicle_info import VehicleInfo


class BaseController(ABC):
    def __init__(self, params: BaseControllerParams, vi: VehicleInfo):
        self.params = params
        self.vi = vi

    @abstractmethod
    def initialize(
        self,
        initial_state: State,
        starting_control_action: ControlAction,
    ):
        pass

    @abstractmethod
    def compute_action(
        self,
        index,
        current_state: State,
        error_state: State,
        trajectory_discretization: TrajectoryDiscretization,
    ) -> (ControlAction, ControllerVizInfo | None):
        pass

    def store_transition(
        self,
        state: State,
        control_action: ControlAction,
        reward: float,
        next_state: State,
        terminated: bool,
    ):
        """
        Can be left empty if controller doesn't need a reward for training
        """
        pass