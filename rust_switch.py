import numpy as np
import scipy

###########################################
# Python Version
###########################################
# from global_planner.spacial_grid import SpatialGrid as SpatialGrid
# from parametric_curves.spiral_optimisation import eval_spiral

# from parametric_curves.spiral_optimisation import optimize_spiral

###########################################
# Rust Version
###########################################
import rust_optimized
from rust_optimized import SpatialGrid as SpatialGrid


def eval_spiral(
    p: tuple[float, float, float, float, float],
    ds: float,
    x_0: float,
    y_0: float,
    psi_0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lists = rust_optimized.eval_spiral(p, x_0, y_0, psi_0, ds)
    return tuple(np.asarray(array) for array in lists)


def optimize_spiral(
    x_0: float,
    y_0: float,
    psi_0: float,
    k_0: float,
    x_f: float,
    y_f: float,
    psi_f: float,
    k_f: float,
    k_max: float,
    equal: bool = False,
) -> tuple[float, float, float, float, float]:
    return rust_optimized.optimize_spiral(
        scipy.optimize, x_0, y_0, psi_0, k_0, x_f, y_f, psi_f, k_f, k_max, equal
    )