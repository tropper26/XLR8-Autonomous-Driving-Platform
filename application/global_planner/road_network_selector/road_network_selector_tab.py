import os
import re

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QScrollArea,
    QSizePolicy,
)
from matplotlib import pyplot as plt

from application.application_status import ApplicationStatus
from application.global_planner.widgets.road_network_canvas import RoadNetworkCanvas
from global_planner.road_network import RoadNetwork
from application.core.flow_layout import FlowLayout


def get_files_with_extension(folder_path, extension):
    files = []

    for file in os.listdir(folder_path):
        if file.endswith(extension):
            files.append(file)

    return files


class RoadNetworkSelectorTab(QWidget):
    def __init__(self, current_app_status: ApplicationStatus, parent=None):
        super().__init__(parent=parent)
        self.layout = QVBoxLayout(self)

        self.current_app_status = current_app_status
        self.graphs_folder = "files/graphs"

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.layout.addWidget(self.scroll_area, stretch=1)

        self.flow_widget = QWidget(self)
        self.flow_widget.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
        )
        self.flow_widget.setStyleSheet(
            f"background-color: {self.current_app_status.color_theme.secondary_color};"
        )
        self.flow_layout = FlowLayout(self.flow_widget)
        self.scroll_area.setWidget(self.flow_widget)

        self.canvases: dict[str, RoadNetworkCanvas] = {}
        self.init_plots()

    def reset(self):
        for canvas in self.canvases.values():
            self.flow_layout.removeWidget(canvas)
            canvas.figure.clear()
            plt.close(canvas.figure)
            canvas.deleteLater()
        self.canvases = {}
        self.init_plots()

    def init_plots(self):
        self.load_canvases()
        self.click_plot_canvas(
            None, self.current_app_status.selected_graph_network_name
        )

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

    def load_canvases(self):
        graph_file_names = get_files_with_extension(self.graphs_folder, ".graphml")

        for graph_file_name in graph_file_names:
            road_network = RoadNetwork(
                file_path=os.path.join(self.graphs_folder, graph_file_name)
            )
            network_graph_canvas = RoadNetworkCanvas(
                road_network_name=self.clean_name(graph_file_name),
                road_network=road_network,
                color_theme=self.current_app_status.color_theme,
                visualize_only=True,
            )

            self.canvases[graph_file_name] = network_graph_canvas

            network_graph_canvas.enterEvent = (
                lambda event, name=graph_file_name: self.enter_plot_canvas(event, name)
            )
            network_graph_canvas.leaveEvent = (
                lambda event, name=graph_file_name: self.leave_plot_canvas(event, name)
            )
            network_graph_canvas.mousePressEvent = (
                lambda event, name=graph_file_name: self.click_plot_canvas(event, name)
            )

            self.flow_layout.addWidget(network_graph_canvas)

        graph_file_name = "Blank Canvas"
        road_network = RoadNetwork(waypoints=[], grid_cell_count=(1, 1))
        network_graph_canvas = RoadNetworkCanvas(
            road_network_name=graph_file_name,
            road_network=road_network,
            color_theme=self.current_app_status.color_theme,
            visualize_only=True,
        )

        self.canvases[graph_file_name] = network_graph_canvas

        network_graph_canvas.enterEvent = (
            lambda event, name=graph_file_name: self.enter_plot_canvas(event, name)
        )
        network_graph_canvas.leaveEvent = (
            lambda event, name=graph_file_name: self.leave_plot_canvas(event, name)
        )
        network_graph_canvas.mousePressEvent = (
            lambda event, name=graph_file_name: self.click_plot_canvas(event, name)
        )

        self.flow_layout.addWidget(network_graph_canvas)

    def click_plot_canvas(self, event, network_graph_name: str):
        # Reset the color of the previously clicked canvas
        currently_selected_figure_canvas = self.canvases[
            self.current_app_status.selected_graph_network_name
        ]
        fig = currently_selected_figure_canvas.figure
        fig.patch.set_facecolor(self.current_app_status.color_theme.background_color)
        fig.canvas.draw()

        # Change the color of the newly clicked canvas
        newly_selected_canvas_with_road_network = self.canvases[network_graph_name]
        fig = newly_selected_canvas_with_road_network.figure

        fig.patch.set_facecolor(self.current_app_status.color_theme.selected_color)
        fig.canvas.draw()

        self.current_app_status.selected_graph_network_name = network_graph_name

    def enter_plot_canvas(self, event, network_graph_name):
        if self.current_app_status.selected_graph_network_name != network_graph_name:
            figure_canvas = self.canvases[network_graph_name]
            figure_canvas.figure.patch.set_facecolor(
                self.current_app_status.color_theme.hover_color
            )
            figure_canvas.figure.canvas.draw()

    def leave_plot_canvas(self, event, network_graph_name):
        if self.current_app_status.selected_graph_network_name != network_graph_name:
            figure_canvas = self.canvases[network_graph_name]
            figure_canvas.figure.patch.set_facecolor(
                self.current_app_status.color_theme.background_color
            )
            figure_canvas.figure.canvas.draw()