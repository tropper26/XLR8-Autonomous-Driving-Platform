from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import scipy

from dto.waypoint import WaypointWithHeading
from global_planner.path_planning.lane_width_infos import LaneWidthInfos
from global_planner.path_planning.path import Path, compute_offset_distances
from parametric_curves.path_segment import PathSegment

from parametric_curves.spiral_input_params import SpiralInputParams
from rust_switch import optimize_spiral
from parametric_curves.spiral_path_segment import SpiralPathSegmentInfo, SpiralPathSegment


def eval_path_segment_concurrently(
        path_segments: list[PathSegment],
    step_size: float,
):
    # def eval_parametric_spiral(parametric_spiral: ParametricCurve[CurveDiscretization], step_size: float):
    #     return parametric_spiral.evaluate(step_size)
    with ThreadPoolExecutor() as executor:
        executor.map(lambda curve: curve.evaluate(step_size), path_segments)


class PathManager:
    def __init__(
        self,
        lane_widths: list[float],
        left_lane_count: int = 0,
        right_lane_count: int = 0,
        k_0: float = 0,
        k_f: float = 0,
        k_max: float = 0.23,
    ):
        self.route: list[WaypointWithHeading] = []
        self.path_segment_input_params: list[SpiralInputParams] = []
        self.path_segments: list[PathSegment] = []
        self.k_0 = k_0
        self.k_f = k_f
        self.k_max = k_max
        self.update_lane_configuration(left_lane_count, right_lane_count, lane_widths)

    def clear(self):
        self.route = []
        self.path_segment_input_params = []
        self.path_segments = []

    def update_lane_configuration(
        self, left_lane_count: int, right_lane_count: int, lane_widths: list[float]
    ):
        self.left_lane_count = left_lane_count
        self.right_lane_count = right_lane_count
        self.lane_widths = lane_widths
        self.lane_offset_distances = compute_offset_distances(
            left_lane_count, right_lane_count, lane_widths
        )

    def compute_segment_params_based_on_start_point_index(self, start_wp_index: int):
        start_waypoint = self.route[start_wp_index]
        end_waypoint = self.route[start_wp_index + 1]

        return SpiralInputParams(
            x_0=start_waypoint.x,
            y_0=start_waypoint.y,
            psi_0=start_waypoint.heading,
            k_0=self.k_0,
            x_f=end_waypoint.x,
            y_f=end_waypoint.y,
            psi_f=end_waypoint.heading,
            k_f=self.k_f,
            k_max=self.k_max,
        )

    def export_path(self) -> Path:
        return Path(
            central_curve_segments=self.path_segments,
            lane_width_infos=LaneWidthInfos(self.left_lane_count, self.right_lane_count, self.lane_widths),
        )

    def create_path_segments_from_route(
        self,
        route: list[WaypointWithHeading],
        step_size: float,
    ) -> list[PathSegment]:
        self.route = route
        segment_params = []
        for i in range(len(route) - 1):
            segment_params.append(
                self.compute_segment_params_based_on_start_point_index(i)
            )

        updated_path_segments = self.optimize_path_segments_concurrently(segment_params)

        eval_path_segment_concurrently(updated_path_segments, step_size)

        self.path_segments = updated_path_segments
        self.path_segment_input_params = segment_params

        return updated_path_segments

    def update_waypoint(
        self,
        modified_route_waypoint: WaypointWithHeading,
        step_size: float,
    ) -> tuple[list[PathSegment], list[int]]:
        index = modified_route_waypoint.index_in_route
        self.route[index] = modified_route_waypoint

        if len(self.route) == 1:
            return [], []

        if index == 0:
            segment_params_to_update = [
                self.compute_segment_params_based_on_start_point_index(index)
            ]
            updated_segments_indexes = [index]
        elif index == len(self.route) - 1:
            segment_params_to_update = [
                self.compute_segment_params_based_on_start_point_index(index - 1)
            ]
            updated_segments_indexes = [index - 1]
        else:
            segment_params_to_update = [
                self.compute_segment_params_based_on_start_point_index(index - 1),
                self.compute_segment_params_based_on_start_point_index(index),
            ]
            updated_segments_indexes = [index - 1, index]

        updated_path_segments = []
        for segment_params in segment_params_to_update:
            path_segment = self.optimize_spiral_path_segment(segment_params)

            updated_path_segments.append(path_segment)

        for updated_segment in updated_path_segments:
            updated_segment.evaluate(step_size)

        for updated_segment_index, updated_segment, segment_params in zip(
            updated_segments_indexes, updated_path_segments, segment_params_to_update
        ):
            self.path_segments[updated_segment_index] = updated_segment
            self.path_segment_input_params[updated_segment_index] = segment_params

        return updated_path_segments, updated_segments_indexes

    def add_first_waypoint(
        self,
        new_waypoint: WaypointWithHeading,
    ):
        self.route = [new_waypoint]


    def insert_waypoint_start(
        self,
        new_waypoint: WaypointWithHeading,
        step_size: float,
    ) -> tuple[PathSegment, int]:

        self.route.insert(0, new_waypoint)
        segment_params = self.compute_segment_params_based_on_start_point_index(0)
        self.path_segment_input_params.insert(0, segment_params)

        optimized_segment = self.optimize_spiral_path_segment(segment_params)

        optimized_segment.evaluate(step_size)
        self.path_segments.insert(0, optimized_segment)

        return optimized_segment, 0

    def append_waypoint(
        self,
        new_waypoint: WaypointWithHeading,
        step_size: float,
    ) -> tuple[PathSegment, int]:
        self.route.append(new_waypoint)

        segment_params = self.compute_segment_params_based_on_start_point_index(-2)
        self.path_segment_input_params.append(segment_params)

        optimized_segment = self.optimize_spiral_path_segment(segment_params)
        optimized_segment.evaluate(step_size)
        self.path_segments.append(optimized_segment)

        return optimized_segment, new_waypoint.index_in_route - 1

    def insert_waypoint(
        self,
        new_waypoint: WaypointWithHeading,
        step_size: float,
    ) -> tuple[list[PathSegment], list[int]]:

        # We are trying to insert a waypoint in the middle of the route so we need 2 new segments
        insertion_index = new_waypoint.index_in_route
        self.route.insert(insertion_index, new_waypoint)

        segment_params_to_update = [
            self.compute_segment_params_based_on_start_point_index(insertion_index - 1),
            self.compute_segment_params_based_on_start_point_index(insertion_index),
        ]
        updated_segments_indexes = [insertion_index - 1, insertion_index]

        updated_path_segments = []
        for segment_params in segment_params_to_update:
            path_segment = self.optimize_spiral_path_segment(segment_params)
            updated_path_segments.append(path_segment)

        for updated_segment in updated_path_segments:
            updated_segment.evaluate(step_size)

        self.path_segments[updated_segments_indexes[0]] = updated_path_segments[0]
        self.path_segment_input_params[updated_segments_indexes[0]] = (
            segment_params_to_update[0]
        )
        self.path_segments.insert(updated_segments_indexes[1], updated_path_segments[1])
        self.path_segment_input_params.insert(
            updated_segments_indexes[1], segment_params_to_update[1]
        )
        return updated_path_segments, updated_segments_indexes

    def add_waypoint(
        self,
        new_waypoint: WaypointWithHeading,
        step_size: float,
    ) -> tuple[list[PathSegment], list[int]]:

        if (
            new_waypoint.index_in_route < 0
            or new_waypoint.index_in_route > len(self.route) + 1
        ):
            raise ValueError("Invalid waypoint index")

        if len(self.route) == 0:
            # If this is the first waypoint, just add it to the route, we can't create a segment yet
            self.add_first_waypoint(new_waypoint)
            return [], []

        if new_waypoint.index_in_route == 0:
            segment, index = self.insert_waypoint_start(new_waypoint, step_size)
            return [segment], [index]

        if new_waypoint.index_in_route == len(self.route):
            segment, index = self.append_waypoint(new_waypoint, step_size)
            return [segment], [index]

        return self.insert_waypoint(new_waypoint, step_size)

    def remove_first_waypoint(
        self,
    ):
        if len(self.route) == 0:
            raise ValueError("Cannot remove first waypoint from an empty route")

        self.route.pop(0)
        if len(self.path_segments) > 0:
            self.path_segments.pop(0)
            self.path_segment_input_params.pop(0)

    def remove_last_waypoint(
        self,
    ):
        if len(self.route) == 0:
            raise ValueError("Cannot remove last waypoint from an empty route")

        self.route.pop(-1)
        self.path_segments.pop(-1)
        self.path_segment_input_params.pop(-1)

    def remove_waypoint_by_index(
        self,
        removed_waypoint_index: int,
        step_size: float,
    ) -> Optional[PathSegment]:

        if removed_waypoint_index < 0 or removed_waypoint_index >= len(self.route):
            raise ValueError("Invalid waypoint index")

        if len(self.route) == 0:
            return None

        if removed_waypoint_index == 0:
            # If the first waypoint is removed, the first segment is removed
            self.remove_first_waypoint()
            return None

        if removed_waypoint_index == len(self.route) - 1:
            # If the last waypoint is removed, the last segment is removed
            self.remove_last_waypoint()
            return None

        # If a waypoint in the middle is removed, two segments are replaced by one
        self.route.pop(removed_waypoint_index)
        self.path_segments.pop(
            removed_waypoint_index
        )  # Remove the segment before the waypoint
        self.path_segment_input_params.pop(
            removed_waypoint_index
        )  # Remove the segment after the waypoint

        # Compute the new segment from the previous waypoint to the new next waypoint
        segment_params = self.compute_segment_params_based_on_start_point_index(
            removed_waypoint_index - 1
        )
        self.path_segment_input_params.insert(
            removed_waypoint_index - 1, segment_params
        )

        optimized_segment = self.optimize_spiral_path_segment(segment_params)
        optimized_segment.evaluate(step_size)
        self.path_segments.insert(removed_waypoint_index - 1, optimized_segment)

        # Return the new segment that replaced the two segments (wp[-1]->wp[+1])
        return optimized_segment

    def optimize_spiral_path_segment(self, segment_params: SpiralInputParams)->PathSegment:
        return SpiralPathSegment(
            spiral_info=SpiralPathSegmentInfo(
                start_point=WaypointWithHeading(
                    x=segment_params.x_0,
                    y=segment_params.y_0,
                    heading=segment_params.psi_0,
                ),
                params=optimize_spiral(*segment_params.as_tuple()),
                offset_distances=self.lane_offset_distances,
            )
        )

    def optimize_path_segments_concurrently(
        self,
        spiral_inputs: list[SpiralInputParams],
    ) -> list[PathSegment]:
        with ThreadPoolExecutor() as executor:
            return list(executor.map(self.optimize_spiral_path_segment, spiral_inputs))