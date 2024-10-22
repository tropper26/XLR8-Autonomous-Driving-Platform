from PyQt5.QtWidgets import QWidget, QVBoxLayout

from application.application_status import ApplicationStatus
from application.global_planner.path_planner.widgets.path_canvas_widget import (
    PathCanvasWidget,
)


class PathPlannerTab(QWidget):
    def __init__(self, current_app_status: ApplicationStatus, parent=None):
        super().__init__(parent=parent)
        self.layout = QVBoxLayout(self)

        self.current_app_status = current_app_status
        self.path_canvas_widget = PathCanvasWidget(current_app_status, parent=self)
        self.last_path_selector_waypoints = None
        self.layout.addWidget(self.path_canvas_widget)

        self.check_for_changes()

    def check_for_changes(self):
        if self.current_app_status.path_selector_waypoints:
            self.path_canvas_widget.init_path_with_waypoints(
                self.current_app_status.path_selector_waypoints
            )
            self.current_app_status.path_selector_waypoints = []