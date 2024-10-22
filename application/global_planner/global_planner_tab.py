from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QTabWidget,
)

from application.application_status import ApplicationStatus
from application.global_planner.path_planner.path_planner_tab import PathPlannerTab
from application.global_planner.path_selector.path_selector_tab import PathSelectorTab
from application.global_planner.road_network_selector.road_network_selector_tab import (
    RoadNetworkSelectorTab,
)
from application.global_planner.trajectory_planner.trajectory_planner_tab import (
    TrajectoryPlannerTab,
)


class GlobalPlannerTab(QWidget):
    def __init__(self, parent, current_app_status: ApplicationStatus):
        super().__init__(parent=parent)
        self.current_app_status = current_app_status

        self.setStyleSheet(
            f"background-color: {current_app_status.color_theme.background_color};"
        )

        self.layout = QHBoxLayout(self)
        self.tab_widget = QTabWidget(self)
        self.layout.addWidget(self.tab_widget)

        # self.tab_widget.setStyleSheet(
        #     f"background-color: {current_app_status.color_theme.primary_color};"
        # )

        self.road_network_selector_tab = RoadNetworkSelectorTab(
            current_app_status=self.current_app_status
        )
        self.tab_widget.addTab(self.road_network_selector_tab, "Road Network Selector")

        self.path_selector_tab = PathSelectorTab(
            current_app_status=self.current_app_status, parent=self
        )
        self.tab_widget.addTab(self.path_selector_tab, "Route Selector")

        self.path_planner_tab = PathPlannerTab(
            current_app_status=self.current_app_status, parent=self
        )
        self.tab_widget.addTab(self.path_planner_tab, "Path Planner")

        self.traj_planner_tab = TrajectoryPlannerTab(
            current_app_status=self.current_app_status,
            parent=self,
        )
        self.tab_widget.addTab(self.traj_planner_tab, "Alternate Paths")

        self.tab_widget.tabBarClicked.connect(self.tab_changed)

    def tab_changed(self, index):
        tab_name = self.tab_widget.tabText(index)
        print(f"Tab changed to: {tab_name}")

        if tab_name == "Road Network Selector":
            self.road_network_selector_tab.reset()
            self.current_app_status.path_selector_waypoints = []
        elif tab_name == "Route Selector":
            self.path_selector_tab.check_for_changes()
        elif tab_name == "Path Planner":
            print("Path Planner tab changed")
            self.current_app_status.delete_ref_path()
            self.path_planner_tab.check_for_changes()
        elif tab_name == "Alternate Paths":
            self.traj_planner_tab.check_for_changes()