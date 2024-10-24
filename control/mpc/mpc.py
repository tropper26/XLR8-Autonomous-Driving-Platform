import numpy as np
import pandas as pd
from qpsolvers import solve_qp

from control.base_controller import BaseController
from control.controller_viz_info import ControllerVizInfo, Types
from control.mpc.cost_function import CostFunction
from control.mpc.mpc_params import MPCParams
from parametric_curves.trajectory import TrajectoryDiscretization
from state_space.augmented_state_space import AugmentedStateSpace
from state_space.inputs.control_action import ControlAction
from state_space.models.kinematic_bicycle_model import KinematicBicycleModel
from state_space.states.state import State
from vehicle.vehicle_info import VehicleInfo


class MPC(BaseController):
    def __init__(
        self,
        params: MPCParams,
        vi: VehicleInfo,
    ):
        super().__init__(params=params, vi=vi)
        self.params = params

        self.aug_state_space = None
        self.delta_U_horizon = None
        self.cost_function = None

    def initialize(
        self,
        initial_state: State,
        starting_control_action: ControlAction,
    ):
        self.initial_state = initial_state.copy()
        self.starting_control_input = starting_control_action.copy()

        self.delta_U_horizon = np.zeros(
            (self.params.horizon_period, self.params.nr_inputs)
        )

        for i in range(self.params.horizon_period):
            self.delta_U_horizon[i] = [
                starting_control_action.d,
                starting_control_action.a,
            ]

        self.aug_state_space = AugmentedStateSpace(
            model=KinematicBicycleModel(vp=self.vi.vp),
            current_state=initial_state.copy(),
            last_input=starting_control_action.copy(),
            sampling_time=self.params.sampling_time,
        )

        self.cost_function = CostFunction(
            C_aug=self.aug_state_space.C_aug,
            vi=self.vi,
            horizon_period=self.params.horizon_period,
            nr_inputs=self.params.nr_inputs,
            weights=self.params.weights,
            sampling_time=self.params.sampling_time,
        )

    def compute_state_over_horizon(self):
        state_space_over_horizon = pd.DataFrame(
            columns=["X_aug", "A_aug", "B_aug"],
            index=range(self.params.horizon_period),
        )
        state_space_over_horizon.loc[0] = [
            self.aug_state_space.X_aug.copy(),
            self.aug_state_space.A_aug.copy(),
            self.aug_state_space.B_aug.copy(),
        ]
        for i in range(1, self.params.horizon_period):
            self.aug_state_space._U = ControlAction(
                a=self.delta_U_horizon[i][0], d=self.delta_U_horizon[i][1]
            )
            self.aug_state_space.propagate_model(self.params.sampling_time)

            state_space_over_horizon.loc[i] = [
                self.aug_state_space.X_aug.copy(),
                self.aug_state_space.A_aug.copy(),
                self.aug_state_space.B_aug.copy(),
            ]

        return state_space_over_horizon

    def compute_action(
        self,
        index,
        current_state: State,
        error_state: State,
        trajectory_discretization: TrajectoryDiscretization,
    ) -> (ControlAction, ControllerVizInfo | None):
        self.aug_state_space.update_based_on_obervations(current_state.copy())

        max_length = len(trajectory_discretization)
        index = 0
        index_horizon = index + self.params.horizon_period
        if index_horizon > max_length:
            index_horizon = max_length
            self.params.horizon_period -= 1

            self.cost_function.update_matrices_for_new_horizon(
                self.params.horizon_period
            )

        # predict the car's movement over the horizon based on the last set of computed control inputs
        state_space_over_horizon = self.compute_state_over_horizon()

        # extract the reference trajectory over the same horizon
        ref_over_horizon = trajectory_discretization[index:index_horizon] #TODO: the rest of this code has not been updated to the new TrajectoryDiscretization API
        print("ref_over_horizon: ", ref_over_horizon)

        # visualize
        aug_states: np.ndarray = state_space_over_horizon.X_aug.values
        X = np.array([aug_state.X for aug_state in aug_states])
        Y = np.array([aug_state.Y for aug_state in aug_states])
        controller_viz = ControllerVizInfo(
            viz_type=Types.Line,
            X=X,
            Y=Y,
            ref_X=ref_over_horizon.X.values,
            ref_Y=ref_over_horizon.Y.values,
        )

        ref_over_horizon = ref_over_horizon.values.reshape(-1, 1)

        # set up cost function to minimize the error over the horizon based on control inputs
        (
            G_horizon,
            Hdb,
            f_T,
            h_horizon,
        ) = self.compute_cost_function_matrices_over_horizon(
            ref_over_horizon, state_space_over_horizon
        )

        # self.display_constraints(G_horizon, h_horizon)

        # minimize the cost function to get the optimal control inputs
        result = self.validate_and_optimize_cost_function(
            G_horizon, Hdb, f_T, h_horizon
        )

        if result is None:
            print("result:", result)
            print(self.delta_U_horizon.T)
            print("Hdb: ", Hdb.shape)
            print("f_T: ", f_T.shape)
            print("G_horizon: ", G_horizon.shape)
            print("h_horizon: ", h_horizon.shape)
        else:
            delta_U_horizon = result.reshape(
                self.params.horizon_period, self.params.nr_inputs
            )
            print("delta_U_horizon: ", delta_U_horizon)
            self.delta_U_horizon = delta_U_horizon

        change_in_control_input = ControlAction(
            a=self.delta_U_horizon[0][0], d=self.delta_U_horizon[0][1]  # type: ignore
        )
        # print("change_in_control_input: ", change_in_control_input)

        self.aug_state_space.update_control_input(change_in_control_input)

        return self.aug_state_space.U, controller_viz

    def validate_and_optimize_cost_function(self, G_horizon, Hdb, f_T, h_horizon):
        is_symmetric = np.allclose(Hdb, Hdb.T, atol=1e-8)
        if not is_symmetric:
            print("Hdb is not symmetric", Hdb.shape)
            print(Hdb - Hdb.T)
            input("Press Enter to continue...")
        return solve_qp(
            Hdb,
            f_T[0],
            # G_horizon,
            # h_horizon.T[0],
            solver="cvxopt",
        )

    def display_constraints(self, G, h):
        """
        Displays the constraints Gx <= h with the specific state variable names.

        Parameters:
        - G: A numpy array of shape (m, n) representing the constraint coefficients.
        - h: A numpy array of shape (m), representing the constraint bounds.
        """
        # Ensure h is a 1D array for consistent indexing
        if h.ndim > 1:
            h = h.flatten()

        variable_names = [
            [f"d_a_{index}", f"d_d_{index}"]
            for index in range(self.params.horizon_period)
        ]
        variable_names = np.array(variable_names).flatten()

        m, n = G.shape
        for i in range(m):
            inequality_parts = []
            for j in range(n):
                if G[i, j] != 0:  # Only add the term if the coefficient is non-zero
                    part = f"{G[i, j]:+.2f}*{variable_names[j]}"
                    inequality_parts.append(part)
            inequality = " + ".join(inequality_parts)
            print(f"{inequality} <= {h[i]:+.2f}")

    def compute_cost_function_matrices_over_horizon(
        self, ref_over_horizon, state_space_over_horizon
    ):
        (
            Hdb,
            Fdb_T,
            G_horizon,
            h_horizon,
        ) = self.cost_function.compute_matrices_over_horizon(state_space_over_horizon)

        f_T = (
            np.concatenate(
                (self.aug_state_space.X_aug.as_column_vector.T, ref_over_horizon.T),
                axis=1,
            )
            @ Fdb_T
        )
        return G_horizon, Hdb, f_T, h_horizon