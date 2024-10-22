import heapq
from typing import Dict, List

from global_planner.road_network import RoadNetwork


class AStar:
    def __init__(self, road_network: RoadNetwork):
        self.road_network = road_network

    def compute_shortest_path(self, start_node_id: int, goal_node_id: int) -> List[int]:
        if (
            start_node_id not in self.road_network.nodes
            or goal_node_id not in self.road_network.nodes
        ):
            raise ValueError("Invalid start or goal node")

        open_set = [(start_node_id, 0.0)]
        came_from: Dict[int, int] = {}

        cost_so_far: Dict[int, float] = {
            node_id: float("inf") for node_id in self.road_network.nodes
        }
        cost_so_far[start_node_id] = 0

        while open_set:
            current_node_id, _ = heapq.heappop(open_set)

            if current_node_id == goal_node_id:
                path = [current_node_id]
                while current_node_id in came_from:
                    current_node_id = came_from[current_node_id]
                    path.insert(0, current_node_id)
                return path

            if self.road_network.successors(
                current_node_id
            ):  # if there are edges from the current node
                for next_node_id in self.road_network.successors(current_node_id):
                    new_cost = cost_so_far[current_node_id] + self.cost_of_edge(
                        current_node_id, next_node_id
                    )
                    if (
                        next_node_id not in cost_so_far
                        or new_cost < cost_so_far[next_node_id]
                    ):
                        cost_so_far[next_node_id] = new_cost
                        priority = new_cost + self.heuristic(next_node_id, goal_node_id)
                        heapq.heappush(open_set, (next_node_id, priority))
                        came_from[next_node_id] = current_node_id

        return []  # No path found

    # def reconstruct_path(self, came_from: Dict[int, int], current: int) -> List[int]:
    #     path = [current]
    #     while current in came_from:
    #         current = came_from[current]
    #         path.insert(0, current)
    #     return path

    def heuristic(self, node_id: int, goal_node_id: int) -> float:
        return self.euclidean_distance(node_id, goal_node_id)

    def cost_of_edge(self, node_id1: int, node_id2: int) -> float:
        return self.euclidean_distance(node_id1, node_id2)

    def euclidean_distance(self, node_id1: int, node_id2: int) -> float:
        node1 = self.road_network.nodes[node_id1]
        node2 = self.road_network.nodes[node_id2]
        return ((node1["x"] - node2["x"]) ** 2 + (node1["y"] - node2["y"]) ** 2) ** 0.5