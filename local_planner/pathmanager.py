from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from dto.waypoint import WaypointWithHeading
from local_planner.parametric_curves.spiral_optimisation import (
    optimize_spiral,
    eval_spiral,
)

SpiralInputParams = namedtuple(
    "SpiralInputParams",
    ["x_0", "y_0", "psi_0", "k_0", "x_f", "y_f", "psi_f", "k_f", "k_max"],
)


class SpiralInfo:
    def __init__(
        self,
        start_point: WaypointWithHeading,
        end_point: WaypointWithHeading,
        k_0: float,
        k_f: float,
        k_max: float,
    ):
        self.start_point = start_point
        self.end_point = end_point
        self.k_0 = k_0
        self.k_f = k_f
        self.k_max = k_max
        self.optim_params = None
        self.df = None

    def as_input_params(self) -> SpiralInputParams:
        return SpiralInputParams(
            x_0=self.start_point.x,
            y_0=self.start_point.y,
            psi_0=self.start_point.heading,
            k_0=self.k_0,
            x_f=self.end_point.x,
            y_f=self.end_point.y,
            psi_f=self.end_point.heading,
            k_f=self.k_f,
            k_max=self.k_max,
        )

    def set_optim_params(self, optim_params):
        self.optim_params = optim_params

    def set_df(self, df):
        self.df = df


class PathManager:
    def __init__(self):
        self.waypoints = None
        self.spiral_lookup = {}
        self.piece_wise_spiral_infos = None
        self.k_0 = 0
        self.k_f = 0
        self.k_max = 1

    def compute_new_path(
        self,
        waypoints: list[WaypointWithHeading],
        step_size: float,
        path_width: float = 0.0,
        return_piece_wise=False,
    ) -> list[pd.DataFrame] | pd.DataFrame:
        """
        Compute a new path based on the waypoints provided

        if path width is provided and is greater than 0, the path will be computed with bounds

        if return_piece_wise is False a single tuple will be returned with the path dataframe, X bounds and Y bounds

        @param waypoints:  List of waypoints to interpolate
        @param step_size:  Step size for the arclength  for the evaluation of the curve [m]
        @param path_width:  Width of the path [m]
        @param return_piece_wise:  Return the curve as a list of dataframes
        @return: piece-wise list of tuples with the path dataframe, X bounds and Y bounds
        """
        if path_width < 0.0:
            raise ValueError("Path width must be greater than or equal 0")

        self.waypoints = waypoints
        new_spiral_lookup = self.get_new_spiral_lookup(waypoints)

        spiral_to_opt_list = list(new_spiral_lookup.values())

        piece_wise_optim_params = self.optimize_spirals_concurrently(spiral_to_opt_list)
        for spiral_info, optim_params in zip(
            spiral_to_opt_list, piece_wise_optim_params
        ):
            spiral_info.set_optim_params(optim_params)

        piece_wise_df = self.eval_spirals_concurrently(spiral_to_opt_list, step_size)
        for spiral_info, df in zip(spiral_to_opt_list, piece_wise_df):
            spiral_info.set_df(df)

        piece_wise_spiral_infos = []
        for i in range(len(waypoints) - 1):
            wp = waypoints[i]
            wp_next = waypoints[i + 1]

            if (wp, wp_next) in new_spiral_lookup:
                piece_wise_spiral_infos.append(new_spiral_lookup[(wp, wp_next)])
            else:
                piece_wise_spiral_infos.append(self.spiral_lookup[(wp, wp_next)])
                new_spiral_lookup[(wp, wp_next)] = self.spiral_lookup[(wp, wp_next)]

        self.spiral_lookup = new_spiral_lookup
        self.piece_wise_spiral_infos = piece_wise_spiral_infos

        piece_wise_spiral_dfs = [
            spiral_info.df for spiral_info in piece_wise_spiral_infos
        ]

        if path_width > 0.0:
            self.compute_curve_bounds(piece_wise_spiral_dfs, path_width / 2)

        if return_piece_wise:
            return piece_wise_spiral_dfs

        return self.concat_spirals(piece_wise_spiral_dfs)

    def compute_new_evaluation(self, step_size: float, path_width: float):
        if self.piece_wise_spiral_infos is None:
            print("compute_new_path must be called before compute_new_evalutation")
            return None
        piece_wise_df = self.eval_spirals_concurrently(
            self.piece_wise_spiral_infos, step_size
        )

        self.compute_curve_bounds(piece_wise_df, path_width / 2)

        return self.concat_spirals(piece_wise_df)

    def get_new_spiral_lookup(
        self, waypoints: list[WaypointWithHeading]
    ) -> dict[(WaypointWithHeading, WaypointWithHeading), SpiralInfo]:
        new_spiral_lookup = {}
        if self.spiral_lookup == {}:
            for i in range(len(waypoints) - 1):
                wp = waypoints[i]
                wp_next = waypoints[i + 1]

                spiral = SpiralInfo(
                    start_point=wp,
                    end_point=wp_next,
                    k_0=self.k_0,
                    k_f=self.k_f,
                    k_max=self.k_max,
                )
                new_spiral_lookup[(wp, wp_next)] = spiral
            return new_spiral_lookup
        for i in range(len(waypoints) - 1):
            wp = waypoints[i]
            wp_next = waypoints[i + 1]

            if (wp, wp_next) not in self.spiral_lookup:
                spiral = SpiralInfo(
                    start_point=wp,
                    end_point=wp_next,
                    k_0=self.k_0,
                    k_f=self.k_f,
                    k_max=self.k_max,
                )
                new_spiral_lookup[(wp, wp_next)] = spiral

        return new_spiral_lookup

    def setup_spirals(
        self, waypoint_pairs: list[(WaypointWithHeading, WaypointWithHeading)]
    ) -> list[SpiralInfo]:
        spirals = []
        for wp, wp_next in waypoint_pairs:
            spiral = SpiralInfo(
                start_point=wp,
                end_point=wp_next,
                k_0=self.k_0,
                k_f=self.k_f,
                k_max=self.k_max,
            )
            spirals.append(spiral)
        return spirals

    def optimize_spirals_concurrently(self, spiral_infos: list[SpiralInfo]):
        with ThreadPoolExecutor() as executor:
            # Map the create_spiral_interpolation function to the parameters
            futures = [
                executor.submit(
                    optimize_spiral,
                    *spiral_info.as_input_params(),
                )
                for spiral_info in spiral_infos
            ]

            piece_wise_optim_params = []

            # Iterate over the futures to get the results in order
            for future in futures:
                p = future.result()  # wait for the result
                piece_wise_optim_params.append(p)

        return piece_wise_optim_params

    def eval_spirals_concurrently(
        self,
        spiral_infos: list[SpiralInfo],
        step_size: float = 0.01,
    ) -> list[pd.DataFrame]:
        with ThreadPoolExecutor() as executor:
            # Map the create_spiral_interpolation function to the parameters
            futures = [
                executor.submit(
                    eval_spiral,
                    spiral_info.optim_params,
                    spiral_info.start_point.x,
                    spiral_info.start_point.y,
                    spiral_info.start_point.heading,
                    step_size,
                )
                for spiral_info in spiral_infos
            ]

            piece_wise_df = []

            # Iterate over the futures to get the results in order
            for future in futures:
                df = future.result()  # wait for the result
                piece_wise_df.append(df)

        return piece_wise_df

    def concat_spirals(self, spiral_dfs: list[pd.DataFrame]) -> pd.DataFrame:
        last_s_value = 0
        for df in spiral_dfs:
            df["S"] += last_s_value
            last_s_value = df["S"].iloc[-1]

        return pd.concat(spiral_dfs, ignore_index=True)

    def compute_curve_bounds(
        self, piece_wise_spiral_dfs: list[pd.DataFrame], distance: float
    ):
        for piece_wise_spiral_df in piece_wise_spiral_dfs:
            x_values = piece_wise_spiral_df["X"].to_numpy()
            y_values = piece_wise_spiral_df["Y"].to_numpy()
            psi_values = piece_wise_spiral_df["Psi"].to_numpy()

            (X_bounds_positive, Y_bounds_positive), (
                X_bounds_negative,
                Y_bounds_negative,
            ) = compute_lateral_points(x_values, y_values, psi_values, distance)

            piece_wise_spiral_df["X_bounds_positive"] = X_bounds_positive
            piece_wise_spiral_df["Y_bounds_positive"] = Y_bounds_positive
            piece_wise_spiral_df["X_bounds_negative"] = X_bounds_negative
            piece_wise_spiral_df["Y_bounds_negative"] = Y_bounds_negative
        return piece_wise_spiral_dfs


def compute_lateral_points(X, Y, Psi, dist):
    # Compute the first derivative (tangent vectors)
    dx_dt = np.cos(Psi)
    dy_dt = np.sin(Psi)

    # Calculate normalized normal vectors
    norms = np.sqrt(dx_dt**2 + dy_dt**2)  # Norm of the tangent vectors
    nx = (
        -dy_dt / norms
    )  # Normal x components (90 degrees rotation of the tangent vector)
    ny = dx_dt / norms

    X_pos = X + nx * dist
    Y_pos = Y + ny * dist
    X_neg = X - nx * dist
    Y_neg = Y - ny * dist

    return (
        (X_pos, Y_pos),
        (X_neg, Y_neg),
    )