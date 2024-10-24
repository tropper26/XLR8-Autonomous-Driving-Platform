class SpatialGrid:
    def __init__(self, min_x: float, max_x: float, min_y: float, max_y: float, grid_cell_count: tuple[float, float]): ...

    def insert_node(self, node_id: int, x: float, y: float):
        pass

    def insert_nodes(self, nodes: list[tuple[int, float, float]]):
        pass

    def remove_node(self, node_id: int, x: float, y: float):
        pass

    def update_node(self, node_id: int, old_x: float, old_y: float, new_x: float, new_y: float) -> int:
        pass

    def get_closest_node(self, x: float, y: float, threshold: float = float("inf")) -> Optional[tuple[int, float, float]]:
        pass
