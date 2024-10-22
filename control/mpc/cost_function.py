import numpy as np
import pandas as pd
from numpy import ndarray

from control.mpc.weights import (
    CostFunctionWeightMatrices,
    Weights,
    OutputWeights,
    InputWeights,
)
from state_space.augmented_state_space import AugmentedStateSpace
from state_space.inputs.control_action import ControlAction
from state_space.states.augmented_state import AugmentedState
from vehicle.vehicle_info import VehicleInfo


class CostFunction:
    def __init__(
        self,
        C_aug: ndarray,
        vi: VehicleInfo,
        horizon_period: int,
        nr_inputs: int,
        sampling_time: float,
        weights: Weights,
    ):
        self.C_aug = C_aug  # this matrix does not vary with time
        self.vi = vi
        self.nr_inputs = nr_inputs
        self.hz = horizon_period

        self.weights: CostFunctionWeightMatrices = self.compute_weight_matrices(
            weights.Q_weights, weights.S_weights, weights.R_weights
        )

        self.CQC_horizon = None
        self.QC_horizon = None
        self.R_horizon = None

        self.C_asterisk_horizon = None

        self.G_U_horizon = None
        self.h_U_horizon = None

        self.update_matrices_for_new_horizon(horizon_period)

    def update_matrices_for_new_horizon(self, horizon_period: int):
        self.hz = horizon_period
        self.compute_cost_function_matrices(self.C_aug)
        self.compute_C_asterisk_horizon()
        self.compute_input_constraint_matrices()

    def compute_matrices_over_horizon(
        self,
        # current_aug_state_space: AugmentedStateSpace,
        # current_delta_U_horizon: ndarray,
        state_space_over_horizon: pd.DataFrame,  # DataFrame with columns: X_aug, A_aug, B_aug
    ):
        Cdb, Adc, G_horizon, h_horizon = self.compute_constraint_matrices(
            state_space_over_horizon
        )

        CQC_Cdb = self.CQC_horizon @ Cdb

        Hdb = Cdb.T @ CQC_Cdb + self.R_horizon

        Fdb_T = np.vstack((Adc.T @ CQC_Cdb, -self.QC_horizon @ Cdb))

        return Hdb, Fdb_T, G_horizon, h_horizon

    def compute_input_constraint_matrices(self):
        """
        | Calculate the linear inequality constraint matrices for the inputs for solving the
          optimization problem using quadpratic programming:
        | G_U_horizon * ΔU <= h_U_horizon
        |
        | Where:
        | -G_U_horizon represents the constraint matrix which is made of two vertically stacked
            block-diagonal matrices of the identity matrix
        | -h_U_horizon represents the constraint bounds for the inputs, which is made of two
            vertically stacked vectors of the maximum and minimum change in control inputs over the
            horizon period
        |
        | :return: None
        """
        # Calculate the number of inputs over the horizon
        nr_inputs_over_horizon = self.nr_inputs * self.hz

        # Create identity matrices for positive and negative vi
        G_U_positive = np.eye(nr_inputs_over_horizon)
        G_U_negative = -np.eye(nr_inputs_over_horizon)

        # Combine the matrices to create the constraint matrix G_U_horizon
        self.G_U_horizon = np.vstack((G_U_positive, G_U_negative))

        max_delta_U = ControlAction(a=self.vi.max_delta_a, d=self.vi.max_delta_d)
        min_delta_U = ControlAction(a=self.vi.min_delta_a, d=self.vi.min_delta_d)

        h_U_positive = np.tile(max_delta_U.as_column_vector, (self.hz, 1))

        h_U_negative = -np.tile(min_delta_U.as_column_vector, (self.hz, 1))

        self.h_U_horizon = np.vstack((h_U_positive, h_U_negative))

    def compute_y_asterisk_min_max(self, X_aug: AugmentedState):
        min_a, max_a = self.vi.compute_acceleration_min_max(X_aug)

        min_y_dot, max_y_dot = self.vi.compute_y_dot_min_max(X_aug.x_dot)

        y_asterisk_min = np.array(
            [
                [self.vi.min_x_dot],
                [min_y_dot],
                [min_a],
                [self.vi.min_d],
            ]
        )

        y_asterisk_max = np.array(
            [
                [self.vi.max_x_dot],
                [max_y_dot],
                [max_a],
                [self.vi.max_d],
            ]
        )

        return y_asterisk_min, y_asterisk_max

    def compute_constraint_matrices(
        self,
        # current_aug_state_space: AugmentedStateSpace,
        # current_delta_U_horizon: ndarray,
        state_space_over_horizon: pd.DataFrame,
    ):
        for index in range(self.hz):
            print(
                f"X_aug[{index}]: {state_space_over_horizon.X_aug[index].X}, {state_space_over_horizon.X_aug[index].Y}, {state_space_over_horizon.X_aug[index].Psi}, {state_space_over_horizon.X_aug[index].x_dot}, {state_space_over_horizon.X_aug[index].y_dot}, {state_space_over_horizon.X_aug[index].psi_dot}, {state_space_over_horizon.X_aug[index].previous_a}, {state_space_over_horizon.X_aug[index].previous_d}"
            )
        print()
        A_aug_horizon = np.zeros((self.hz, *state_space_over_horizon.A_aug[0].shape))
        B_aug_horizon = np.zeros((self.hz, *state_space_over_horizon.B_aug[0].shape))

        y_asterisk_min, y_asterisk_max = self.compute_y_asterisk_min_max(
            state_space_over_horizon.X_aug[0]
        )

        y_asterisk_min_horizon = np.array(y_asterisk_min)
        y_asterisk_max_horizon = np.array(y_asterisk_max)

        # variable that stores the cumulative product of A matrices over the horizon
        A_cumulative_product = state_space_over_horizon.A_aug[0]

        Adc = np.array(A_cumulative_product)

        for index in range(self.hz):
            A_aug_horizon[index] = state_space_over_horizon.A_aug[index]
            B_aug_horizon[index] = state_space_over_horizon.B_aug[index]

            if index < self.hz - 1:
                A_cumulative_product = (
                    state_space_over_horizon.A_aug[index + 1] @ A_cumulative_product
                )

                Adc = np.vstack((Adc, A_cumulative_product))

                (
                    y_asterisk_min,
                    y_asterisk_max,
                ) = self.compute_y_asterisk_min_max(
                    state_space_over_horizon.X_aug[index + 1]
                )

                y_asterisk_min_horizon = np.vstack(
                    (y_asterisk_min_horizon, y_asterisk_min)
                )
                y_asterisk_max_horizon = np.vstack(
                    (y_asterisk_max_horizon, y_asterisk_max)
                )

        Cdb = self.compute_Cdb(A_aug_horizon, B_aug_horizon, self.hz)

        C_asterisk_CDB_horizon = self.C_asterisk_horizon @ Cdb

        G_Y_horizon = np.vstack((C_asterisk_CDB_horizon, -C_asterisk_CDB_horizon))

        # transpose to row vector and extract the row so that it is a 1D array
        C_asterisk_horizon_Adc_X_aug_K = (
            self.C_asterisk_horizon
            @ Adc
            @ state_space_over_horizon.X_aug[0].as_column_vector
        )

        y_asterisk_max_horizon_difference = (
            y_asterisk_max_horizon - C_asterisk_horizon_Adc_X_aug_K
        )

        y_asterisk_min_horizon_difference = (
            -y_asterisk_min_horizon + C_asterisk_horizon_Adc_X_aug_K
        )

        h_Y_horizon = np.vstack(
            (y_asterisk_max_horizon_difference, y_asterisk_min_horizon_difference)
        )

        G_horizon = np.vstack((self.G_U_horizon, G_Y_horizon))
        h_horizon = np.vstack((self.h_U_horizon, h_Y_horizon))

        return Cdb, Adc, G_horizon, h_horizon

    @staticmethod
    def compute_Cdb(
        A_aug_horizon: ndarray, B_aug_horizon: ndarray, horizon_period: int
    ):
        """
        | Computes the C double bar (Cdb) matrix which together with the A cumulative product (ACP) matrix
        | can be used to predict the state over the horizon period.
        |
        | for horizon_period = 4, Cdb :
        | [ B0   0    0    0
        |   A1B0 B1   0    0
        |   A2B0 A2B1 B2   0
        |   A3B0 A3B1 A3B2 B3 ]

        :param A_aug_horizon: 3-dimensional array containing the A matrices over the horizon period
        :param B_aug_horizon: 3-dimensional array containing the B matrices over the horizon period
        :param horizon_period: number of time steps to predict over
        """
        B_rows, B_cols = B_aug_horizon[0].shape
        Cdb = np.zeros((B_rows * horizon_period, B_cols * horizon_period), dtype=float)

        for col_index in range(horizon_period):
            cumulative_product = B_aug_horizon[col_index]

            start_col = B_cols * col_index
            end_col = B_cols * (col_index + 1)
            start_row = B_rows * col_index
            end_row = B_rows * (col_index + 1)

            # Fill in the diagonal
            Cdb[start_row:end_row, start_col:end_col] = cumulative_product

            # Fill in the lower triangle
            for row_index in range(col_index + 1, horizon_period):
                cumulative_product = A_aug_horizon[row_index] @ cumulative_product
                start_row = B_rows * row_index
                end_row = B_rows * (row_index + 1)

                Cdb[start_row:end_row, start_col:end_col] = cumulative_product

        return Cdb

    def compute_C_asterisk_horizon(self):
        """
        Computes the C asterisk matrix over the horizon period which extracts the relevant states from the augmented
        state vector to apply vi.

        @return: 2-dimensional array containing the C asterisk matrix over the horizon period
        """
        # 2-dimensional array that extracts the relevant states from the augmented state vector for vi
        C_asterisk = np.array(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )

        # Create a block-diagonal matrix C_asterisk_horizon for vi
        self.C_asterisk_horizon = np.kron(np.eye(self.hz, dtype=float), C_asterisk)

    @staticmethod
    def compute_weight_matrices(
        Q_weights: OutputWeights,
        S_weights: OutputWeights,
        R_weights: InputWeights,
    ):
        """
        Create Weight Matrices for Model Predictive Control (MPC)

        This function creates weight matrices used in Model Predictive Control (MPC)
        to optimize a cost function over a time horizon. The weight matrices Q and S
        affect the outputs of the system, while R affects the inputs.

        :param Q_weights: Weight coefficients for Q matrix affecting the first horizon-1 outputs.
        :type Q_weights: OutputWeights

        :param S_weights: Weight coefficients for S matrix affecting the final output.
        :type S_weights: OutputWeights

        :param R_weights: Weight coefficients for R matrix affecting the inputs.
        :type R_weights: InputWeights

        """
        Q = np.diag(
            [
                Q_weights.X_global,
                Q_weights.Y_global,
                Q_weights.heading_angle,
                Q_weights.velocity,
            ]
        )
        S = np.diag(
            [
                S_weights.X_global,
                S_weights.Y_global,
                S_weights.heading_angle,
                S_weights.velocity,
            ]
        )
        R = np.diag([R_weights.acceleration, R_weights.steering_angle])

        return CostFunctionWeightMatrices(Q, S, R)

    def compute_cost_function_matrices(self, C_aug: ndarray):
        """
        | Calculate block-diagonal matrices for the cost function in Model Predictive Control (MPC).
        |
        | This function calculates and returns three block-diagonal matrices used in MPC cost functions:
        | 1. State Cost Weight Matrix (SCW):  Represents the cost or penalty associated with state variable tracking or
                                              regulation in the MPC cost function
        | 2. Reference Tracking Weight Matrix (RTW): Represents the weight or penalty associated with tracking the
                                                     reference trajectory for state variables in the MPC cost function
        | 3. Input Cost Weight Matrix (ICW): Represents the cost or penalty associated with control input effort or
                                               manipulation in the MPC cost function
        |
        | In MPC, the cost function is typically defined as follows:
        |
        | J = 1/2 * X_aug_h.T * SCW * X_aug_h - ref_h.T * RTW * X_aug_h + 1/2 * ΔU.T * ICW * ΔU
        |
        | Where:
        | - X_aug_h: The augmented state vector predicted over the horizon.
        | - ref_h: The reference state trajectory, representing the desired state at each time step over the horizon.
        | - ΔU: The change in control inputs being optimized.
        |
        | :param model: StateSpace object representing the system model.
        | :param weights: CostFunctionWeightMatrices object containing weight matrices (Q, S, and R).
        | :param horizon_period: Integer specifying the prediction horizon.
        """

        # Calculate CQC, CSC, QC, and SC matrices
        CQC: ndarray = C_aug.T @ self.weights.Q @ C_aug
        CSC: ndarray = C_aug.T @ self.weights.S @ C_aug
        QC: ndarray = self.weights.Q @ C_aug
        SC: ndarray = self.weights.S @ C_aug

        # Create a block-diagonal matrix SPM for State Cost weights
        self.CQC_horizon = np.kron(np.eye(self.hz, dtype=float), CQC)
        self.CQC_horizon[
            -CSC.shape[0] :, -CSC.shape[1] :
        ] = CSC  # Last diagonal position with CSC

        # Create a block-diagonal matrix OPM for Reference Tracking weights
        self.QC_horizon = np.kron(np.eye(self.hz, dtype=float), QC)
        self.QC_horizon[
            -SC.shape[0] :, -SC.shape[1] :
        ] = SC  # Last diagonal position with SC

        # Create a block-diagonal matrix IPM for Input Cost weights
        self.R_horizon = np.kron(np.eye(self.hz, dtype=float), self.weights.R)