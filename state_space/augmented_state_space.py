from state_space.inputs.control_action import ControlAction
from state_space.models.generic_model import GenericModel
from state_space.state_space import StateSpace
from state_space.state_space_matrices import StateSpaceMatrices
from state_space.states.augmented_state import AugmentedState
from state_space.states.state import State


class AugmentedStateSpace(StateSpace):
    """
    The augmented state space model of the vehicle.
    """

    def __init__(
        self,
        model: GenericModel,
        current_state: State,
        last_input: ControlAction,
        sampling_time: float,
    ):
        super().__init__(model, current_state, last_input, sampling_time)

        self._X = AugmentedState(current_state, last_input)
        self._M: StateSpaceMatrices = self.model.compute_augmented_matrices(self._M)

    @property
    def X_aug(self) -> AugmentedState:
        return self._X

    @property
    def X(self) -> State:
        return self.X_aug.as_state()

    @property
    def U(self) -> ControlAction:
        return self.X_aug.prev_U

    @property
    def delta_U(self):
        return self._U

    @property
    def A_aug(self):
        return self._M.A

    @property
    def B_aug(self):
        return self._M.B

    @property
    def C_aug(self):
        return self._M.C

    @property
    def D_aug(self):
        return self._M.D

    def generate_state_space_matrices(self, elapsed_time: float) -> StateSpaceMatrices:
        """Compute the continous linear parameter varying state space augmented model

        | X_aug(k+1) = A_aug(k) * X_aug(k) + B_aug(k) * ΔU(k)
        | Y_aug(k)   = C_aug(k) * X_aug(k) + D_aug(k) * ΔU(k)

        :param elapsed_time: time elapsed since last iteration
        :return: the augmented state vector and the state space matrices
        """

        discrete_matrices = super().generate_state_space_matrices(elapsed_time)

        return self.model.compute_augmented_matrices(discrete_matrices)

    def update_based_on_obervations(self, observed_state: State):
        """
        Update the state space form with the new state of the vehicle based on data from the sensors

        @param observed_state: the observed (maybe noisy) state of the vehicle
        """
        self._X = AugmentedState(observed_state, self._X.prev_U)

        self._M = self.generate_state_space_matrices(self.sampling_time)

    def update_control_input(self, delta_U: ControlAction):
        """
        Update the state space form with the new change in control input

        @param delta_U: the new change in control input
        """
        self._U = delta_U

        new_U = self.X_aug.prev_U + self.delta_U

        self._X = AugmentedState(self.X, new_U)