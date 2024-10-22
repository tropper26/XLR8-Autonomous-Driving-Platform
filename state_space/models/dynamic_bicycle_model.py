import numpy as np

from numpy import cos, sin

from state_space.inputs.control_action import ControlAction
from state_space.models.generic_model import GenericModel
from state_space.state_space_matrices import StateSpaceMatrices

from state_space.states.state import State
from vehicle.vehicle_params import VehicleParams


class DynamicBicycleModel(GenericModel):
    def __init__(self, vp: VehicleParams):
        super().__init__(vp=vp)

    def _compute_continuous_matrices(
        self, X: State, U: ControlAction
    ) -> StateSpaceMatrices:
        x_dot, y_dot, Psi = X.x_dot, X.y_dot, X.Psi
        delta = U.d
        m, mju, Cf, Cr, lf, lr, Iz, GRAVITY = (
            self.vp.m,
            self.vp.mju,
            self.vp.Cf,
            self.vp.Cr,
            self.vp.lf,
            self.vp.lr,
            self.vp.Iz,
            self.vp.GRAVITY,
        )
        # print("State: ", X, "Control: ", U)
        u = 3  # offset to avoid division by zero

        A14 = cos(Psi)
        A15 = -sin(Psi)

        A24 = sin(Psi)
        A25 = cos(Psi)

        A44 = -mju * GRAVITY / (x_dot + u)
        A45 = Cf * sin(delta) / (m * (x_dot + u))
        A46 = (Cf * lf * sin(delta) / (m * (x_dot + u))) + y_dot

        A55 = -(Cr + Cf * cos(delta)) / (m * (x_dot + u))
        A56 = ((Cr * lr) - (Cf * lf * cos(delta))) / (m * (x_dot + u)) - x_dot

        A65 = ((Cr * lr) - (Cf * lf * cos(delta))) / (Iz * (x_dot + u))
        A66 = -((Cf * lf**2 * cos(delta)) + (Cr * lr**2)) / (Iz * (x_dot + u))
        # print(
        #     f"A66 = {Cf} * {lf}^2 * {cos(delta)} + {Cr} * {lr}^2 / {Iz} * ({x_dot} + {u}) = {A66}"
        # )

        A = np.array(
            [
                [0.0, 0.0, 0.0, A14, A15, 0.0],
                [0.0, 0.0, 0.0, A24, A25, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, A44, A45, A46],
                [0.0, 0.0, 0.0, 0.0, A55, A56],
                [0.0, 0.0, 0.0, 0.0, A65, A66],
            ]
        )

        B42 = -(1 / m) * sin(delta) * Cf
        B52 = (1 / m) * cos(delta) * Cf
        B62 = (1 / Iz) * cos(delta) * Cf * lf

        B = np.array(
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, B42], [0.0, B52], [0.0, B62]]
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

    def compute_discrete_matrices(
        self, X: State, U: ControlAction, delta_t: float
    ) -> StateSpaceMatrices:
        """
        | Euler's discretization of a linear parameter varying state space model matrices.
        |
        | Continous linear parameter varying state space model:
        | X(t)   = A*X(t-1) + B*U(t-1)
        | Y(t)   = C*X(t-1) + D*U(t-1)
        |
        | Discrete linear parameter varying state space model:
        | ( X(k+1)-X(k) ) / Δt = X(k+1)
        | ( X(k+1)-X(k) ) / Δt = A*X(k) + B*U(k)
        | ( X(k+1)-X(k) )      = Δt*A*X(k) + Δt*B*U(k)
        |   X(k+1)             = Δt*A*X(k) + Δt*B*U(k) + X(k)
        |   X(k+1)             = (I + Δt*A)*X(k) + Δt*B*U(k)
        | => A_d = I + Δt*A
        | => B_d = Δt*B
        |
        | C and D are not affected by time discretization:
        | => C_d = C
        | => D_d = D
        |
        | X(k+1) = A_d*X(k) + B_d*U(k)
        | Y(k)   = C_d*X(k) + D_d*U(k)
        |
        :param delta_t: time step
        :return: A_d, B_d, C_d, D_d discretized state space matrices
        """
        continuous_matrices = self._compute_continuous_matrices(X, U)

        A_d = (
            np.eye(np.size(continuous_matrices.A, 1)) + continuous_matrices.A * delta_t
        )
        B_d = continuous_matrices.B * delta_t
        C_d = continuous_matrices.C
        D_d = continuous_matrices.D

        return StateSpaceMatrices(A_d, B_d, C_d, D_d)

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