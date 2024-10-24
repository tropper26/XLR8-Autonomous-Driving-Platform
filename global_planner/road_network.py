from typing import List, Optional

import networkx as nx
from scipy.spatial import distance

from dto.waypoint import Waypoint, WaypointWithHeading
from rust_switch import SpatialGrid


def is_point_to_left(start: Waypoint, end: Waypoint, point: Waypoint) -> bool:
    vector = (end.x - start.x, end.y - start.y)
    relative_point = (point.x - start.x, point.y - start.y)
    cross_product = vector[0] * relative_point[1] - relative_point[0] * vector[1]
    return cross_product > 0


def point_at_distance(
    vector_start: Waypoint, vector_end: Waypoint, distance: float
) -> Waypoint:
    vector_direction = (
        vector_end.x - vector_start.x,
        vector_end.y - vector_start.y,
    )
    vector_magnitude = (vector_direction[0] ** 2 + vector_direction[1] ** 2) ** 0.5
    unit_vector = (
        vector_direction[0] / vector_magnitude,
        vector_direction[1] / vector_magnitude,
    )

    new_point = Waypoint(
        vector_end.x + unit_vector[0] * distance,
        vector_end.y + unit_vector[1] * distance,
    )

    return new_point


class RoadNetwork:
    LANE_WIDTH = 0.35

    def __init__(
        self,
        file_path: str = None,
        waypoints: List[WaypointWithHeading] = None,
        grid_cell_count: tuple[int, int] = (32, 32),
    ):
        if waypoints is None and file_path is None:
            raise Exception("Either waypoints or file_path must be provided")
        if waypoints is not None:
            self.G = nx.DiGraph()
            for i, waypoint in enumerate(waypoints):
                self.G.add_node(i, x=waypoint.x, y=waypoint.y, heading=waypoint.heading)
                if i > 0:
                    self.G.add_edge(i - 1, i)
        else:
            self.G = self.read_graphml(file_path)
        self.grid = None
        self.init_grid(grid_cell_count)

    def add_world_info(self, world_x_lim, world_y_lim, lane_width, lane_count):
        self.G.graph["world_x_lim"] = world_x_lim
        self.G.graph["world_y_lim"] = world_y_lim
        self.G.graph["lane_width"] = lane_width
        self.G.graph["lane_count"] = lane_count

    def get_world_info(self):
        if self.G.graph.get("world_x_lim") is None:
            return None, None, None, None
        return (
            self.G.graph["world_x_lim"],
            self.G.graph["world_y_lim"],
            self.G.graph["lane_width"],
            self.G.graph["lane_count"],
        )

    def init_grid(self, grid_cell_count):
        """
        Initialize the spatial grid for the graph. This is used for faster coordinate based nearest neighbor queries.
        """
        if self.G.graph.get("x_min") is None:
            min_x, max_x, min_y, max_y = self.calculate_bounds()
            self.G.graph["x_min"] = min_x
            self.G.graph["x_max"] = max_x
            self.G.graph["y_min"] = min_y
            self.G.graph["y_max"] = max_y
        else:
            min_x = self.G.graph["x_min"]
            max_x = self.G.graph["x_max"]
            min_y = self.G.graph["y_min"]
            max_y = self.G.graph["y_max"]
        self.grid = SpatialGrid(min_x, max_x, min_y, max_y, grid_cell_count)

        for node_id, data in self.G.nodes(data=True):
            self.grid.insert_node(int(node_id), data["x"], data["y"])

    @property
    def nodes(self):
        return self.G.nodes

    @property
    def edges(self):
        return self.G.edges

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            self.G.graph["x_min"],
            self.G.graph["x_max"],
            self.G.graph["y_min"],
            self.G.graph["y_max"],
        )

    def successors(self, node_id: int) -> List[int]:
        return list(self.G.successors(node_id))

    def update_node_position(self, node_id: int, new_x: float, new_y: float):
        self.grid.update_node(
            node_id,
            self.G.nodes[node_id]["x"],
            self.G.nodes[node_id]["y"],
            new_x,
            new_y,
        )
        self.G.nodes[node_id]["x"] = new_x
        self.G.nodes[node_id]["y"] = new_y

    def get_closest_node(
        self, x: float, y: float, threshold: float = float("inf")
    ) -> Optional[tuple[int, float, float]]:
        """
        Get the closest node to the given coordinates (x, y) within a specified threshold.
        Returns a tuple (node_id, node_x, node_y) if found, otherwise None.
        """
        if self.grid is None:
            raise Exception("Grid not initialized")
        return self.grid.get_closest_node(x, y, threshold)

    def get_all_waypoints(self) -> List[Waypoint]:
        return [
            Waypoint(data["x"], data["y"]) for node, data in self.G.nodes(data=True)
        ]

    def get_waypoints_based_on_ids(
        self, node_ids: List[int]
    ) -> List[WaypointWithHeading]:
        if not node_ids:
            return []
        return [
            WaypointWithHeading(
                self.G.nodes[node_id]["x"],
                self.G.nodes[node_id]["y"],
                self.G.nodes[node_id]["heading"]
                if "heading" in self.G.nodes[node_id]
                else 0,
            )
            for node_id in node_ids
        ]

    def calculate_bounds(self):
        min_x = min_y = float("inf")
        max_x = max_y = float("-inf")

        for _, data in self.G.nodes(data=True):
            x, y = data["x"], data["y"]
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)

        return min_x, max_x, min_y, max_y

    def get_grouped_waypoints(self, threshold_radius=0.1) -> List[List[int]]:
        grouped_waypoints = []
        visited = set()

        for node_id, waypoint in self.G.nodes(data=True):  # i
            if node_id not in visited:
                group = [node_id]
                visited.add(node_id)

                for other_id, other_waypoint in self.G.nodes(data=True):
                    if (
                        other_id not in visited
                        and distance.euclidean(
                            (waypoint["x"], waypoint["y"]),
                            (other_waypoint["x"], other_waypoint["y"]),
                        )
                        <= threshold_radius
                    ):
                        group.append(other_id)
                        visited.add(other_id)

                grouped_waypoints.append(group)

        return [group for group in grouped_waypoints if len(group) > 1]

    def fix_intersections(self):
        intersection_groups = self.get_grouped_waypoints()
        for intersection_group in intersection_groups:
            for intersection_node in intersection_group:
                predecessors = list(self.G.predecessors(intersection_node))
                if len(predecessors) > 1:
                    raise Exception("Intersection node has more than one predecessor")
                predecessor_id = predecessors[0]
                prepredecessors = list(self.G.predecessors(predecessor_id))
                if len(prepredecessors) != 1:
                    raise Exception("Predecessor node has more than one predecessor")
                prepredecessor_id = prepredecessors[0]

                successors = list(self.G.successors(intersection_node))
                relevant_coord_index = (
                    0
                    if abs(
                        self.G.nodes[prepredecessor_id]["x"]
                        - self.G.nodes[predecessor_id]["x"]
                    )
                    > abs(
                        self.G.nodes[prepredecessor_id]["y"]
                        - self.G.nodes[predecessor_id]["y"]
                    )
                    else 1
                )

                for successor_id in successors:
                    successor = self.G.nodes[successor_id]
                    difference = (
                        successor["x" if relevant_coord_index == 0 else "y"]
                        - self.G.nodes[predecessor_id][
                            "x" if relevant_coord_index == 0 else "y"
                        ]
                    )
                    if abs(difference) < RoadNetwork.LANE_WIDTH:
                        forward_successor = successor_id
                    elif is_point_to_left(
                        Waypoint(
                            self.G.nodes[predecessor_id]["x"],
                            self.G.nodes[predecessor_id]["y"],
                        ),
                        Waypoint(
                            self.G.nodes[intersection_node]["x"],
                            self.G.nodes[intersection_node]["y"],
                        ),
                        Waypoint(successor["x"], successor["y"]),
                    ):
                        left_successor = successor_id
                    else:
                        right_successor = successor_id

                new_point = point_at_distance(
                    Waypoint(
                        self.G.nodes[prepredecessor_id]["x"],
                        self.G.nodes[prepredecessor_id]["y"],
                    ),
                    Waypoint(
                        self.G.nodes[predecessor_id]["x"],
                        self.G.nodes[predecessor_id]["y"],
                    ),
                    0.5,
                )
                self.G.nodes[intersection_node]["x"] = new_point.x
                self.G.nodes[intersection_node]["y"] = new_point.y

    def write_graphml(self, file_path: str):
        nx.write_graphml(self.G, file_path)

    def read_graphml(self, file_path: str):
        return nx.read_graphml(file_path, node_type=int)