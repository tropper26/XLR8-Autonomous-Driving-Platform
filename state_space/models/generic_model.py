from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from state_space.inputs.control_action import ControlAction
from state_space.state_space_matrices import StateSpaceMatrices
from state_space.states.base_state import BaseState
from vehicle.vehicle_params import VehicleParams

GenericState = TypeVar("GenericState", bound=BaseState)


class GenericModel(ABC, Generic[GenericState]):
    def __init__(self, vp: VehicleParams):
        self.vp = vp

    @abstractmethod
    def _compute_continuous_matrices(
        self, X: GenericState, U: ControlAction
    ) -> StateSpaceMatrices:
        """
        | Compute the continous linear parameter varying state space model matrices.
        |
        | Continous linear parameter varying state space model:
        | X(t)   = A*X(t-1) + B*U(t-1)
        | Y(t)   = C*X(t-1) + D*U(t-1)
        |
        :return: A, B, C, D state space matrices
        """
        pass

    @abstractmethod
    def compute_discrete_matrices(
        self, X: GenericState, U: ControlAction, delta_t: float
    ) -> StateSpaceMatrices:
        pass

    @abstractmethod
    def compute_augmented_matrices(
        self, discrete_matrices: StateSpaceMatrices
    ) -> StateSpaceMatrices:
        pass