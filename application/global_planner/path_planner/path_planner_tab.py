from PyQt5.QtWidgets import QWidget, QVBoxLayout

from application.application_status import ApplicationStatus
from application.global_planner.path_planner.widgets.path_canvas_widget import (
    PathPlannerWidget,
)


class PathPlannerTab(QWidget):
    def __init__(self, current_app_status: ApplicationStatus, parent=None):
        super().__init__(parent=parent)
        self.layout = QVBoxLayout(self)

        self.current_app_status = current_app_status
        self.path_planner_widget = PathPlannerWidget(current_app_status, parent=self)
        self.last_path_selector_waypoints = None
        self.layout.addWidget(self.path_planner_widget)

        self.check_for_changes()

    def check_for_changes(self):
        if self.current_app_status.selected_route:
            self.path_planner_widget.init_path_from_route(
                self.current_app_status.selected_route
            )
            self.current_app_status.selected_route = []