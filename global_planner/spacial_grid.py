from typing import Optional

import numpy as np
from sortedcontainers import SortedList


class SpatialGrid:
    def __init__(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        grid_cell_count: tuple[int, int],
    ):
        self.min_x = min_x
        self.min_y = min_y
        self.cell_size_x = (max_x - min_x) / grid_cell_count[0] + 1e-6
        self.cell_size_y = (max_y - min_y) / grid_cell_count[1] + 1e-6
        self.grid: dict[tuple[int, int], SortedList[tuple[float, float, int]]] = {}

    def grid_key(self, x: float, y: float) -> tuple[int, int]:
        return (
            int((x - self.min_x) / self.cell_size_x),
            int((y - self.min_y) / self.cell_size_y),
        )

    def insert_node(self, node_id: int, x: float, y: float):
        key = self.grid_key(x, y)
        if key not in self.grid:
            self.grid[key] = SortedList()
        self.grid[key].add((x, y, node_id))

    def insert_nodes(self, nodes: list[tuple[int, float, float]]):
        for node_id, x, y in nodes:
            self.insert_node(node_id, x, y)

    def remove_node(self, node_id: int, x: float, y: float):
        key = self.grid_key(x, y)
        if key in self.grid:
            self.grid[key].remove((x, y, node_id))
            if len(self.grid[key]) == 0:
                del self.grid[key]

    def update_node(
        self, node_id: int, old_x: float, old_y: float, new_x: float, new_y: float
    ):
        self.remove_node(node_id, old_x, old_y)
        self.insert_node(node_id, new_x, new_y)

    def get_closest_node(
        self, x: float, y: float, threshold: float = float("inf")
    ) -> Optional[tuple[int, float, float]]:
        """
        Get the closest node to the given coordinates (x, y) within a specified threshold.
        Returns a tuple (node_id, node_x, node_y) if found, otherwise None.

        If threshold is set to infinity, the search will be exhaustive, finding the closest node in the grid no
        matter the distance. Else, the search will be limited to a square of side length 2*threshold centered at the
        given coordinates.
        """
        center_key = self.grid_key(x, y)
        min_distance = threshold
        closest_node = None

        if threshold == float("inf"):
            # If threshold is infinite, examine every node in the grid
            all_keys = self.grid.keys()
        else:
            range_search = max(
                int(threshold / self.cell_size_x) + 1,
                int(threshold / self.cell_size_y) + 1,
            )
            all_keys = (
                (center_key[0] + dx, center_key[1] + dy)
                for dx in range(-range_search, range_search + 1)
                for dy in range(-range_search, range_search + 1)
            )

        for key in all_keys:
            if key in self.grid:
                for nx, ny, node_id in self.grid[key]:
                    dist = np.sqrt((nx - x) ** 2 + (ny - y) ** 2)
                    if dist < min_distance:
                        min_distance = dist
                        closest_node = (node_id, nx, ny)

        return closest_node if closest_node and min_distance <= threshold else None