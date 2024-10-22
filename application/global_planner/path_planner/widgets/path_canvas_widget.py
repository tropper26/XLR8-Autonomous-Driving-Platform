import time

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QLineEdit,
    QLabel,
    QButtonGroup,
    QRadioButton,
    QSpacerItem,
)

from application.application_status import ApplicationStatus
from application.global_planner.path_planner.widgets.path_canvas import (
    PathCanvas,
    DrawMode,
)
from dto.geometry import Rectangle
from dto.waypoint import WaypointWithHeading
from global_planner.road_network import RoadNetwork


class PathCanvasWidget(QWidget):
    def __init__(self, current_app_status: ApplicationStatus, parent=None):
        super().__init__(parent=parent)
        self.graph_network_name = None
        self.current_app_status = current_app_status

        self.layout = QHBoxLayout(self)
        self.path_canvas = PathCanvas(
            world_x_limit=self.current_app_status.world_x_limit,
            world_y_limit=self.current_app_status.world_y_limit,
            color_theme=current_app_status.color_theme,
            parent=self,
        )

        self.path_canvas.pathParamsChanged.connect(self.path_params_changed)

        self.layout.addWidget(self.path_canvas, stretch=5)

        self.settings_widget = QWidget(self)
        self.settings_layout = QVBoxLayout(self.settings_widget)
        self.layout.addWidget(self.settings_widget, stretch=1)

        self.settings_widget.setStyleSheet(
            """
              QLabel {
                  font-size: 14px;
                  color: White;
                  border: none;
                  background-color: #9c9999;
                  font-weight: bold;
              }
              QLineEdit, QRadioButton {
                  border: 2px solid darkgray;
                  border-radius: 4px;
                  padding: 10px;
                  background-color: #9c9999;
              }
              QLineEdit:focus, QRadioButton:focus {
                  border-color: red;
              }

            """
        )

        self.add_control_fields()
        self.settings_layout.addStretch(1)

    def setup_mode_selection(self):
        self.mode_widget = QWidget(self.settings_widget)
        self.mode_layout = QVBoxLayout(self.mode_widget)
        self.settings_layout.addWidget(self.mode_widget)

        self.mode_label = QLabel("Select Mode:")
        self.mode_label.setStyleSheet(f"background-color: #9c9999; padding: 10px;")
        self.mode_layout.addWidget(self.mode_label)

        self.mode_button_group = QButtonGroup(self.settings_widget)
        for mode in DrawMode:
            radio_button = QRadioButton(mode.name.replace("_", " ").title())
            self.mode_button_group.addButton(radio_button)
            self.mode_layout.addWidget(radio_button)
            radio_button.toggled.connect(
                lambda checked, m=mode: self.set_mode(m) if checked else None
            )
            if mode == DrawMode.Add_Waypoint:  # Set the default selected mode
                radio_button.setChecked(True)

    def add_control_fields(self):
        self.add_field(
            "Network Name", "Enter network name", slot=self.update_network_name
        )

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_graph)
        self.save_button.setStyleSheet(
            f"background-color: {self.current_app_status.color_theme.primary_color}; "
            + f"color: {self.current_app_status.color_theme.button_text_color};"
            + f"padding: 10px 20px; border: none; border-radius: 5px; font-size: 14px;"
        )
        self.settings_layout.addWidget(self.save_button)

        self.settings_layout.addSpacerItem(QSpacerItem(0, 20))

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_path_canvas)
        self.clear_button.setStyleSheet(
            f"background-color: {self.current_app_status.color_theme.primary_color}; "
            + f"color: {self.current_app_status.color_theme.button_text_color};"
            + f"padding: 10px 20px; border: none; border-radius: 5px; font-size: 14px;"
        )
        self.settings_layout.addWidget(self.clear_button)

        self.settings_layout.addSpacerItem(QSpacerItem(0, 20))
        self.setup_mode_selection()
        self.settings_layout.addSpacerItem(QSpacerItem(0, 20))

        self.add_field(
            "Simulation X Limit",
            "Enter X limit",
            value=self.current_app_status.world_x_limit,
            slot=self.update_x_limit,
        )
        self.add_field(
            "Simulation Y Limit",
            "Enter Y limit",
            value=self.current_app_status.world_y_limit,
            slot=self.update_y_limit,
        )
        self.add_field(
            "Lane Width",
            "Enter lane width",
            value=self.current_app_status.lane_width,
            slot=self.update_lane_width,
        )
        self.add_field(
            "Number of Lanes",
            "Enter number of lanes",
            value=self.current_app_status.lane_count,
            slot=self.update_lane_count,
        )

    def add_field(self, label_text, placeholder, value=None, slot=None):
        self.field_widget = QWidget(self.settings_widget)
        self.field_layout = QVBoxLayout(self.field_widget)
        self.field_widget.setStyleSheet(f"background-color: #9c9999; padding: 10px;")
        self.settings_layout.addWidget(self.field_widget)
        label = QLabel(label_text)
        if value is None:
            line_edit = QLineEdit()
        else:
            line_edit = QLineEdit(str(value))
        line_edit.setPlaceholderText(placeholder)
        if slot:
            line_edit.textChanged.connect(slot)
        self.field_layout.addWidget(label)
        self.field_layout.addWidget(line_edit)

    def set_mode(self, mode):
        self.path_canvas.mode = mode
        print(f"Mode changed to: {mode.name.title()}")

    @pyqtSlot(str, name="update_network_name")
    def update_network_name(self, value):
        self.graph_network_name = value.strip()

    @pyqtSlot(str, name="update_x_limit")
    def update_x_limit(self, value):
        try:
            x_limit = float(value)
            if x_limit <= 0:
                raise ValueError
            self.current_app_status.world_x_limit = x_limit
            self.path_canvas.recalculate_positions(
                self.current_app_status.world_x_limit,
                self.current_app_status.world_y_limit,
            )
        except ValueError:
            pass  # Handle or log error appropriately

    @pyqtSlot(str, name="update_y_limit")
    def update_y_limit(self, value):
        try:
            y_limit = float(value)
            if y_limit <= 0:
                raise ValueError
            self.current_app_status.world_y_limit = y_limit
            self.path_canvas.recalculate_positions(
                self.current_app_status.world_x_limit,
                self.current_app_status.world_y_limit,
            )
        except ValueError:
            pass  # Handle or log error appropriately

    @pyqtSlot(str, name="update_path_width")
    def update_lane_width(self, value):
        try:
            lane_width = float(value)
            if lane_width <= 0:
                raise ValueError
            self.current_app_status.lane_width = lane_width
            self.path_canvas.emit_new_path_params()
        except ValueError:
            pass  # Handle or log error appropriately

    @pyqtSlot(str, name="update_lanes_in_path")
    def update_lane_count(self, value):
        try:
            lane_count = int(value)
            if lane_count <= 0:
                raise ValueError
            self.current_app_status.lane_count = lane_count
            self.path_canvas.emit_new_path_params()
        except ValueError:
            pass  # Handle or log error appropriately

    def init_path_with_waypoints(self, waypoints: list[WaypointWithHeading]):
        self.path_canvas.world_x_limit = self.current_app_status.world_x_limit
        self.path_canvas.world_y_limit = self.current_app_status.world_y_limit
        self.path_canvas.init_path(waypoints)

    def clear_path_canvas(self):
        self.path_canvas.clear_all()
        self.current_app_status.delete_ref_path()

    def save_graph(self):
        name = self.graph_network_name
        if not name:
            print("Network name cannot be empty")
            return
        road_network = RoadNetwork(
            waypoints=self.path_canvas.waypoints_with_headings,
        )
        road_network.add_world_info(
            self.current_app_status.world_x_limit,
            self.current_app_status.world_y_limit,
            self.current_app_status.lane_width,
            self.current_app_status.lane_count,
        )
        road_network.write_graphml(f"files/graphs/%{name}.graphml")

    def path_params_changed(
        self, path_params: (list[WaypointWithHeading], list[Rectangle])
    ):
        start_time = time.time()
        self.current_app_status.delete_ref_path()

        waypoints_with_heading, obstacles = path_params
        self.current_app_status.path_obstacles = obstacles

        if len(waypoints_with_heading) >= 2:
            piece_wise_spiral_dfs = (
                self.current_app_status.path_manager.compute_new_path(
                    waypoints_with_heading,
                    step_size=self.current_app_status.path_visualisation_step_size,
                    path_width=self.current_app_status.path_width,
                    return_piece_wise=True,
                )
            )
            print(
                f"Time taken to compute path in milliseconds: {(time.time() - start_time) * 1000}"
            )
            self.path_canvas.update_bounds(piece_wise_spiral_dfs)
            end_time = time.time()
            print(
                f"Time taken to compute+viz path in milliseconds: {(end_time - start_time) * 1000}"
            )