from collections import namedtuple
from typing import TypeVar, Generic

from state_space.inputs.control_action import ControlAction
from state_space.models.generic_model import GenericModel
from state_space.state_space_matrices import StateSpaceMatrices
from state_space.states.base_state import BaseState
from state_space.states.state import State

AccelerationState = namedtuple(
    "AccelerationState", ["x_dot_dot", "y_dot_dot", "psi_dot_dot"]
)

GenericState = TypeVar("GenericState", bound=BaseState)


class StateSpace(Generic[GenericState]):
    """
    The augmented state space form of the vehicle.
    """

    def __init__(
        self,
        model: GenericModel[GenericState],
        current_state: GenericState,
        last_input: ControlAction,
        sampling_time: float,
    ):
        self.model = model
        self._X = current_state
        self._U = last_input
        self.sampling_time = sampling_time

        self._M: StateSpaceMatrices = self.model.compute_discrete_matrices(
            self._X, self._U, sampling_time
        )

    @property
    def X(self):
        return self._X

    @property
    def U(self):
        return self._U

    @property
    def A(self):
        return self._M.A

    @property
    def B(self):
        return self._M.B

    @property
    def C(self):
        return self._M.C

    @property
    def D(self):
        return self._M.D

    @property
    def state_size(self):
        return self._X.size

    @property
    def input_size(self):
        return self._U.size

    def generate_state_space_matrices(self, elapsed_time: float) -> StateSpaceMatrices:
        """Generate the discrete linear parameter varying state space model

        | X(k+1) = A(k) * X(k) + B(k) * ΔU(k)
        | Y(k)   = C(k) * X(k) + D(k) * ΔU(k)


        :param elapsed_time: time elapsed since last iteration
        :return: StateSpaceMatrices
        """
        discrete_matrices = self.model.compute_discrete_matrices(
            self.X, self.U, elapsed_time
        )

        return discrete_matrices

    def update_based_on_obervations(self, observed_state: State):
        """
        Update the state space form with the new state of the vehicle based on data from the sensors
        ( control inputs shouldn't have changed in the time step )
        @param observed_state: the observed (maybe noisy) state of the vehicle
        """
        self._X = observed_state

        self._M = self.generate_state_space_matrices(self.sampling_time)

    def update_control_input(self, U: ControlAction):
        """
        Update the state space form with the new control input

        @param U: the new control input
        """
        self._U = U

    def propagate_model(self, elapsed_time: float, sub_sampling_multiplier: int = 1):
        """
        Propagate the model one time step forward by computing the movement the dynamic object
        would suffer based on the current state new control inputs.

        @param elapsed_time: time elapsed since last iteration
        @param sub_sampling_multiplier: the number of times
            the model is propagated in the time step ( higher = more accurate )

        """
        # if sub_sampling_multiplier < 1:
        #     raise ValueError("Sub sampling multiplier must be greater or equal to 1")
        #
        # if sub_sampling_multiplier > 1:
        #     self._U = self._U / sub_sampling_multiplier
        #     elapsed_time = elapsed_time / sub_sampling_multiplier
        #
        # for i in range(sub_sampling_multiplier):
        self._X.as_column_vector = (
            self.A @ self._X.as_column_vector + self.B @ self._U.as_column_vector
        )

        self._M = self.generate_state_space_matrices(elapsed_time)