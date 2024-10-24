from concurrent.futures import ThreadPoolExecutor

import numpy as np

from parametric_curves.path_segment import PathSegmentDiscretization, PathSegment

def check_and_print_array(arr: np.ndarray | list, decimals: int = 2) -> None:
    if isinstance(arr, list):
        arr = np.array(arr)

    if arr.ndim == 1:
        formatted_arr = np.round(arr, decimals=decimals)
        formatted_row = "\t".join(f"{x:.{decimals}f}" for x in formatted_arr)
        print(formatted_row)

    elif arr.ndim == 2:
        formatted_arr = np.round(arr, decimals=decimals)
        rows, cols = formatted_arr.shape

        def format_row(row):
            return "\t".join(f"{x:.{decimals}f}" for x in row)

        if rows > 10:
            # Print first 5 and last 5 rows if more than 10 rows
            for row in formatted_arr[:5]:
                print(format_row(row))
            print("...")
            for row in formatted_arr[-5:]:
                print(format_row(row))
        else:
            # Print all rows if 10 or fewer
            for row in formatted_arr:
                print(format_row(row))
    else:
        print("The array is neither 1D nor 2D.")


def lane_offsets(
        start_idx: int, lane_count: int, direction: int, lane_widths: list[float], central_lane_half_width: float
) -> list[float]:
    """
    Calculates the offset for lanes on one side (either left or right).
    Offsets are calculated cumulatively for each lane.
    """
    offsets = []
    cumulative_offset = central_lane_half_width  # Start at the center

    for i in range(lane_count):
        lane_width = lane_widths[start_idx + i]

        # Shoulder offset (before lane)
        offsets.append(direction * cumulative_offset)

        # Central line offset of the lane
        cumulative_offset += lane_width / 2
        offsets.append(direction * cumulative_offset)

        # End of lane offset (after lane)
        cumulative_offset += lane_width / 2
        offsets.append(direction * cumulative_offset)

    return offsets


def compute_offset_distances(
        left_lane_count: int, right_lane_count: int, lane_widths: list[float]
) -> list[float]:
    """
    Calculates the offset distances for the central lines and shoulders of all lanes,
    ensuring that overlapping shoulders aren't duplicated.
    """
    central_lane_half_width = lane_widths[left_lane_count] / 2

    distances = []

    # Handle the right side, including shoulder if no lanes
    if right_lane_count > 0:
        # Calculate distances for right lanes (negative offsets)
        right_distances = lane_offsets(left_lane_count, right_lane_count, -1, lane_widths, central_lane_half_width)
        distances += right_distances
    else:
        right_shoulder = -central_lane_half_width
        distances.append(right_shoulder)

    # Add central lane at offset 0 (for the main lane)
    distances.append(0)

    # Handle the left side, including shoulder if no lanes
    if left_lane_count > 0:
        # Calculate distances for left lanes (positive offsets)
        left_distances = lane_offsets(0, left_lane_count, 1, lane_widths, central_lane_half_width)
        distances += left_distances
    else:
        left_shoulder = central_lane_half_width
        distances.append(left_shoulder)

    return distances


def concat_path_segment_discretizations(
        discretizations: list[PathSegmentDiscretization]) -> PathSegmentDiscretization:

    X = np.concatenate([disc.X for disc in discretizations])
    Y = np.concatenate([disc.Y for disc in discretizations])
    Psi = np.concatenate([disc.Psi for disc in discretizations])
    K = np.concatenate([disc.K for disc in discretizations])

    for i in range(1, len(discretizations)):
        discretizations[i].S += discretizations[i - 1].S[-1]

    S = np.concatenate([disc.S for disc in discretizations])

    lateral_X = np.concatenate([disc.lateral_X for disc in discretizations])
    lateral_Y = np.concatenate([disc.lateral_Y for disc in discretizations])

    path_disc = PathSegmentDiscretization(S, X, Y, Psi, K)
    path_disc.lateral_X = lateral_X
    path_disc.lateral_Y = lateral_Y

    return path_disc


class Path:
    def __init__(
            self,
            central_curve_segments: list[PathSegment],
    ):
        self.central_curve_segments = central_curve_segments
        self.discretized = concat_path_segment_discretizations(
            [segment.discretized for segment in central_curve_segments]
        )

    # def create_path_discretization

    def reevaluate_path_discretization(self, step_size: float):
        with ThreadPoolExecutor() as executor:
            executor.map(lambda curve: curve.evaluate(step_size), self.central_curve_segments)

        self.discretized = concat_path_segment_discretizations(
            [segment.discretized for segment in self.central_curve_segments]
        )

    def update_lane_configuration(self, left_lane_count: int, right_lane_count: int, lane_widths: list[float]):
        lane_offset_distances = compute_offset_distances(
            left_lane_count, right_lane_count, lane_widths
        )

        # Update the offset distances for all curve segments for future discretization
        for segment in self.central_curve_segments:
            segment.offset_distances = lane_offset_distances

        # Recompute the lateral points for the current path discretization
        self.discretized.compute_lateral_points(lane_offset_distances)