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
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.grid_cell_count = grid_cell_count
        self.cell_size_x = (max_x - min_x) / grid_cell_count[0] + 1e-6
        self.cell_size_y = (max_y - min_y) / grid_cell_count[1] + 1e-6
        self.grid = {}

    def grid_key(self, x: float, y: float) -> tuple[int, int]:
        return (
            int((x - self.min_x) / self.cell_size_x),
            int((y - self.min_y) / self.cell_size_y),
        )

    def insert_node_into_grid(self, node_id: int, x: float, y: float):
        key = self.grid_key(x, y)
        if key not in self.grid:
            self.grid[key] = SortedList()
        self.grid[key].add((x, y, node_id))

    def remove_node_from_grid(self, node_id: int, x: float, y: float):
        key = self.grid_key(x, y)
        if key in self.grid:
            self.grid[key].remove((x, y, node_id))
            if len(self.grid[key]) == 0:
                del self.grid[key]

    def update_node_in_grid(
        self, node_id: int, old_x: float, old_y: float, new_x: float, new_y: float
    ):
        self.remove_node_from_grid(node_id, old_x, old_y)
        self.insert_node_into_grid(node_id, new_x, new_y)

    def get_closest_node(
        self, x: float, y: float, threshold: float = float("inf")
    ) -> Optional[tuple[int, float, float]]:
        """
        Get the closest node to the given coordinates (x, y) within a specified threshold.
        Returns a tuple (node_id, node_x, node_y) if found, otherwise None.
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