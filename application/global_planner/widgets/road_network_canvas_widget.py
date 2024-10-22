import os
import re

from PyQt5.QtWidgets import QWidget, QVBoxLayout

from application.application_status import ApplicationStatus
from application.global_planner.widgets.road_network_canvas import RoadNetworkCanvas
from global_planner.road_network import RoadNetwork


class RoadNetworkCanvasWidget(QWidget):
    def __init__(self, current_app_status: ApplicationStatus, parent=None):
        super().__init__(parent=parent)
        self.layout = QVBoxLayout(self)
        self.graphs_folder = "files/graphs"
        self.current_app_status = current_app_status

        self.current_graph_network_name = current_app_status.selected_graph_network_name
        self.current_path_node_ids: list[str] = []
        self.network_graph_canvas = None

    def clean_name(self, input_string):
        # Replace underscores with spaces first
        input_string = input_string.replace("_", " ")

        # Remove everything after the last dot to handle extensions
        input_string = re.sub(r"\.[^.]*$", "", input_string)

        # Remove all non-alphanumeric characters (except spaces)
        input_string = re.sub(r"[^a-zA-Z0-9\s]", "", input_string)

        # Normalize whitespace (convert multiple spaces to one, trim leading/trailing spaces)
        input_string = re.sub(r"\s+", " ", input_string).strip()
        return input_string

    def check_selected_network_changed(self):
        if (
            self.current_graph_network_name
            != self.current_app_status.selected_graph_network_name
        ):
            if self.network_graph_canvas is not None:
                self.network_graph_canvas.deleteLater()
                self.layout.removeWidget(self.network_graph_canvas)

            self.current_graph_network_name = (
                self.current_app_status.selected_graph_network_name
            )
            self.road_network = RoadNetwork(
                file_path=os.path.join(
                    self.graphs_folder, self.current_graph_network_name
                )
            )

            if self.current_graph_network_name.startswith("%"):
                self.current_path_node_ids: list[int] = sorted(
                    [int(node) for node in self.road_network.nodes()]
                )  # sort as integers
                self.current_path_node_ids = [
                    str(node) for node in self.current_path_node_ids
                ]

                self.network_graph_canvas = RoadNetworkCanvas(
                    road_network_name=self.clean_name(self.current_graph_network_name),
                    road_network=self.road_network,
                    color_theme=self.current_app_status.color_theme,
                    visualize_only=False,
                    initial_path=self.current_path_node_ids,
                )
            else:
                self.network_graph_canvas = RoadNetworkCanvas(
                    road_network_name=self.clean_name(self.current_graph_network_name),
                    road_network=self.road_network,
                    color_theme=self.current_app_status.color_theme,
                    visualize_only=False,
                )
            self.network_graph_canvas.path_nodes_Changed.connect(self.set_selected_path)

            self.layout.addWidget(self.network_graph_canvas)
        self.set_selected_path(self.current_path_node_ids)

    def set_selected_path(self, path_node_ids: list[str]):
        self.current_path_node_ids = path_node_ids
        self.current_app_status.path_selector_waypoints = (
            self.road_network.get_waypoints_based_on_ids(path_node_ids)
        )
        (
            world_x_lim,
            world_y_lim,
            lane_width,
            lane_count,
        ) = self.road_network.get_world_info()
        if world_x_lim is None:
            x_min, x_max, y_min, y_max = self.road_network.bounds
            self.current_app_status.world_x_limit = 1.2 * x_max
            self.current_app_status.world_y_limit = 1.2 * y_max
        else:
            self.current_app_status.world_x_limit = world_x_lim
            self.current_app_status.world_y_limit = world_y_lim
            self.current_app_status.lane_width = lane_width
            self.current_app_status.lane_count = lane_count