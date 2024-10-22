from typing import List

from global_planner.astar import AStar
from global_planner.road_network import RoadNetwork


class GlobalPlanner:
    def __init__(self):
        pass

    def find_shortest_path(
        self, road_network: RoadNetwork, start_node_id: int, goal_node_id: int
    ) -> List[int]:
        aStar = AStar(road_network)
        return aStar.compute_shortest_path(start_node_id, goal_node_id)

    def find_shortest_path_with_stops(
        self, road_network: RoadNetwork, nodes_to_visit: List[int]
    ) -> List[int]:
        aStar = AStar(road_network)
        path = []
        for i in range(len(nodes_to_visit) - 1):
            if len(path) > 0:
                path.pop()  # Remove last element, since it is the same as the first element of the next path

            sub_path = aStar.compute_shortest_path(
                nodes_to_visit[i], nodes_to_visit[i + 1]
            )

            if len(sub_path) == 0:
                print(
                    f"No path found between {nodes_to_visit[i]} and {nodes_to_visit[i + 1]}"
                )
                return []
            path += sub_path

        return path