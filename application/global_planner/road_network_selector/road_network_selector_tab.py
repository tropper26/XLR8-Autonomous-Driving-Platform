import os
import re

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QScrollArea,
    QSizePolicy,
)

from application.application_status import ApplicationStatus
from application.global_planner.widgets.road_network_image import RoadNetworkImage
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
        self.images_folder = "files/images"

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

        self.road_network_images: dict[str, RoadNetworkImage] = {}
        self.init_plots()

    def reset(self):
        for image in self.road_network_images.values():
            self.flow_layout.removeWidget(image)

        self.road_network_images = {}
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
            road_network_image = RoadNetworkImage(
                road_network_name=self.clean_name(graph_file_name),
                path_to_image=os.path.join(self.images_folder, graph_file_name.replace(".graphml", ".png")),
                path_to_no_figure_found=os.path.join(self.images_folder, "no_figure_found.png"),
                color_theme=self.current_app_status.color_theme
            )

            road_network_image.setFixedSize(300,300)

            road_network_image.enterEvent = (
                lambda event, name=graph_file_name: self.enter_plot_canvas(event, name)
            )

            road_network_image.leaveEvent = (
                lambda event, name=graph_file_name: self.leave_plot_canvas(event, name)
            )

            road_network_image.mousePressEvent = (
                lambda event, name=graph_file_name: self.click_plot_canvas(event, name)
            )

            self.road_network_images[graph_file_name] = road_network_image

            self.flow_layout.addWidget(road_network_image)

        blank_graph_file_name = "Blank Canvas"

        blank_road_network_image = RoadNetworkImage(
            road_network_name=blank_graph_file_name,
            path_to_image=os.path.join(self.images_folder, "blank_canvas.png"),
            path_to_no_figure_found=os.path.join(self.images_folder, "no_figure_found.png"),
            color_theme=self.current_app_status.color_theme
        )

        blank_road_network_image.setFixedSize(300,300)

        blank_road_network_image.enterEvent = (
            lambda event, name=blank_graph_file_name: self.enter_plot_canvas(event, name)
        )

        blank_road_network_image.leaveEvent = (
            lambda event, name=blank_graph_file_name: self.leave_plot_canvas(event, name)
        )

        blank_road_network_image.mousePressEvent = (
            lambda event, name=blank_graph_file_name: self.click_plot_canvas(event, name)
        )

        self.road_network_images[blank_graph_file_name] = blank_road_network_image

        self.flow_layout.addWidget(blank_road_network_image)

    def click_plot_canvas(self, event, network_graph_name: str):
        # Reset the color of the previously clicked canvas
        currently_selected_road_network = self.road_network_images[
            self.current_app_status.selected_graph_network_name
        ]

        currently_selected_road_network.text_label.setStyleSheet(
            f"color: {self.current_app_status.color_theme.button_text_color};"
            f"font-size: 20px;"
        )
        currently_selected_road_network.pixmap_label.setStyleSheet(
            f"border: 8px;"
            f"border-style: solid;"
            f"border-color: {self.current_app_status.color_theme.secondary_color};"
            f"border-radius: 10px;"
        )

        # Change the color of the newly clicked canvas
        road_network_image = self.road_network_images[network_graph_name]
        road_network_image.text_label.setStyleSheet(
            f"color: {self.current_app_status.color_theme.hover_color};"
            f"font-size: 20px;"
        )
        road_network_image.pixmap_label.setStyleSheet(
            f"border: 8px;"
            f"border-style: solid;"
            f"border-color: {self.current_app_status.color_theme.hover_color};"
            f"border-radius: 10px;"
        )

        self.current_app_status.selected_graph_network_name = network_graph_name

    def enter_plot_canvas(self, event, network_graph_name):
        if self.current_app_status.selected_graph_network_name != network_graph_name:
            road_network_image = self.road_network_images[network_graph_name]
            road_network_image.pixmap_label.setStyleSheet(
                f"border: 8px;"
                f"border-style: solid;"
                f"border-color: {self.current_app_status.color_theme.background_color};"
                f"border-radius: 10px;"
            )

    def leave_plot_canvas(self, event, network_graph_name):
        if self.current_app_status.selected_graph_network_name != network_graph_name:
            road_network_image = self.road_network_images[network_graph_name]
            road_network_image.pixmap_label.setStyleSheet(
                f"border: 8px;"
                f"border-style: solid;"
                f"border-color: {self.current_app_status.color_theme.secondary_color};"
                f"border-radius: 10px;"
            )
