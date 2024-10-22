from state_space.inputs.control_action import ControlAction
from state_space.models.generic_model import GenericModel
from state_space.state_space_matrices import StateSpaceMatrices
from state_space.states.state import State
import numpy as np
from numpy import cos, sin, tan

from vehicle.vehicle_params import VehicleParams


class KinematicBicycleModel(GenericModel):
    def __init__(self, vp: VehicleParams):
        super().__init__(vp=vp)

    def _compute_continuous_matrices(
        self, X: State, U: ControlAction
    ) -> StateSpaceMatrices:
        pass

    def compute_discrete_matrices(
        self, X: State, U: ControlAction, delta_t: float
    ) -> StateSpaceMatrices:
        dt = delta_t
        u = 0.001  # offset to avoid division by zero
        A14 = cos(X.Psi) * dt
        A15 = -sin(X.Psi) * dt
        A24 = sin(X.Psi) * dt
        A25 = A14
        A61 = ((X.x_dot**2 + X.y_dot**2) ** 0.5 * tan(U.d)) / (
            self.vp.wheelbase * (X.X + u)
        )
        A = np.array(
            [
                [1.0, 0.0, 0.0, A14, A15, 0.0],
                [0.0, 1.0, 0.0, A24, A25, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, dt],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [A61, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        B11 = (dt * dt) / 2

        B = np.array(
            [
                [B11, 0],
                [0.0, 0.0],
                [0.0, 0.0],
                [dt, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        C = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ]
        )

        D = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

        return StateSpaceMatrices(A, B, C, D)

    def compute_augmented_matrices(
        self,
        discrete_matrices: StateSpaceMatrices,
    ) -> StateSpaceMatrices:
        """
        Augment the lpv state space model with the previous inputs, which results in new Matrices
        | [ X(k+1) ] = [ A_d B_d ] * [ X(k) ] + [ B_d ] * [ ΔU(k) ]
        | [ U(k)   ] = [ 0   I   ] * [ U(k) ] + [ I   ] * [ ΔU(k) ]
        | \\=======/    \=========/   \======/   \=====/   \=======/
        | X_aug(k+1)     A_aug       X_aug(k)    B_aug    change in U
        |
        | [ Y(k) ]   = [ C_d  0  ] * [ X(k) U(k-1) ].T + [ D_d  I  ] * [ ΔU(k) ]
        | \======/     \=========/   \============/      \=========/   \=======/
        | Y_aug(k)      C_aug          X_aug(k)           D_aug       change in U
        :param A_d: Discretised Matrix that relates the state at time k+1 to the state at time k
        :param B_d: Discretised Matrix that relates the state at time k+1 to the input at time k
        :param C_d: Discretised Matrix that extracts the output from the state
        :param D_d: Discretised Matrix that extracts the output from the input (usually zero)
        :return: A_aug, B_aug, C_aug, D_aug augmented state space matrices
        """
        O = np.zeros((discrete_matrices.B.shape[1], discrete_matrices.A.shape[0]))
        I = np.identity(discrete_matrices.B.shape[1])

        # Stack A and B horizontally (along columns)
        AB_stacked = np.hstack((discrete_matrices.A, discrete_matrices.B))

        # Stack O and I horizontally (along columns)
        OI_stacked = np.hstack((O, I))

        # Stack AB_stacked and OI_stacked vertically (along rows)
        A_aug = np.vstack((AB_stacked, OI_stacked))
        B_aug = np.vstack((discrete_matrices.B, I))

        O = np.zeros((discrete_matrices.C.shape[0], discrete_matrices.B.shape[1]))

        C_aug = np.hstack((discrete_matrices.C, O))
        D_aug = discrete_matrices.D

        return StateSpaceMatrices(A_aug, B_aug, C_aug, D_aug)