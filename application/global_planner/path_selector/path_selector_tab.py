from PyQt5.QtWidgets import QWidget, QVBoxLayout

from application.application_status import ApplicationStatus
from application.global_planner.widgets.road_network_canvas_widget import (
    RoadNetworkCanvasWidget,
)


class PathSelectorTab(QWidget):
    def __init__(self, current_app_status: ApplicationStatus, parent=None):
        super().__init__(parent=parent)
        self.layout = QVBoxLayout(self)

        self.current_app_status = current_app_status
        self.road_network_widget = RoadNetworkCanvasWidget(
            current_app_status, parent=self
        )

        self.layout.addWidget(self.road_network_widget)

        self.check_for_changes()

    def check_for_changes(self):
        if self.current_app_status.selected_graph_network_name == "Blank Canvas":
            self.current_app_status.selected_route = []
            self.current_app_status.world_x_limit = (
                self.current_app_status.initial_world_x_limit
            )
            self.current_app_status.world_y_limit = (
                self.current_app_status.initial_world_y_limit
            )
            self.road_network_widget.hide()
        else:
            self.road_network_widget.check_selected_network_changed()
            self.road_network_widget.show()