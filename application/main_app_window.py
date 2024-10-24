import sys
from enum import Enum

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QMainWindow,
    QTabWidget,
)
from matplotlib import pyplot as plt

from application.application_status import ApplicationStatus
from application.core.dynamic_tab_widget import DynamicTabWidget
from application.global_planner.global_planner_tab import GlobalPlannerTab
from application.simulation_runner.simulation_runner_tab import SimulationRunnerTab
from application.simulation_setup.simulation_setup_tab import SimulationSetupTab
from application.visualizer.visualizer_tab import VisualizerTab


class MainTabNames(Enum):
    GLOBAL_PLANNER = "Global Planning"
    SIMULATION_SETUP = "Setup Simulations"
    SIMULATION_RUNNER = "Run Simulations"
    VISUALIZER = "Visualize Simulations"


class MainAppWindow(QMainWindow):
    def __init__(self, current_app_status: ApplicationStatus):
        super().__init__()
        self.simulation_setup_widget = None
        self.setWindowTitle("Autonomous Vehicle Simulator")
        self.resize(1700, 850)
        self.current_app_status = current_app_status
        self.fullscreen = False

        plt.style.use(
            {
                "axes.facecolor": current_app_status.color_theme.background_color,
                "figure.facecolor": current_app_status.color_theme.background_color,
                "axes.edgecolor": "white",
                "axes.labelcolor": "white",
                "text.color": current_app_status.color_theme.text_color,
                "xtick.color": "white",
                "ytick.color": "white",
            }
        )

        self.central_widget = QWidget(parent=self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(20, 20, 0, 0)  # Adjust margins
        self.central_widget.setLayout(self.layout)

        self.tab_widget = QTabWidget(self)
        self.current_tab_index = self.tab_widget.currentIndex()
        self.layout.addWidget(self.tab_widget)

        self.global_planner_tab = GlobalPlannerTab(self, self.current_app_status)
        self.tab_widget.addTab(
            self.global_planner_tab, MainTabNames.GLOBAL_PLANNER.value
        )

        self.simulations_tabs = DynamicTabWidget(
            tabs_factory=self.add_simulation,
            tabs_name="Simulation",
            color_theme=self.current_app_status.color_theme,
            parent=self,
        )
        self.tab_widget.addTab(
            self.simulations_tabs, MainTabNames.SIMULATION_SETUP.value
        )

        self.simulation_runner_tab = SimulationRunnerTab(self, self.current_app_status)
        self.tab_widget.addTab(
            self.simulation_runner_tab, MainTabNames.SIMULATION_RUNNER.value
        )

        self.visualizer_tab = VisualizerTab(self, self.current_app_status)
        self.tab_widget.addTab(self.visualizer_tab, MainTabNames.VISUALIZER.value)

        self.tab_widget.tabBarClicked.connect(self.tab_changed)

        self.previous_tab_name = MainTabNames.GLOBAL_PLANNER.value

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F11:
            if self.fullscreen:
                self.showNormal()
            else:
                self.showFullScreen()
            self.fullscreen = not self.fullscreen
        elif event.key() == Qt.Key_Escape:
            if self.fullscreen:
                self.showNormal()
                self.fullscreen = False

    def add_simulation(self):
        self.current_app_status.add_simulation()

        self.simulation_setup_widget = SimulationSetupTab(
            current_app_status=self.current_app_status,
            simulation_index=self.current_app_status.simulation_count,
            parent=self,
        )

        return self.simulation_setup_widget

    def tab_changed(self, new_tab_index):
        new_tab_name = self.tab_widget.tabText(new_tab_index)
        print(f"Tab changed to: {new_tab_name}")

        if (
                self.previous_tab_name == MainTabNames.GLOBAL_PLANNER.value
                and new_tab_name != MainTabNames.GLOBAL_PLANNER.value
        ):
            self.current_app_status.ref_path = (
                self.global_planner_tab.path_planner_tab.path_planner_widget.path_canvas.path_manager.export_path()
            )  # TODO This is a hack, fix this :O

        if (
                new_tab_name == MainTabNames.SIMULATION_RUNNER.value
                and self.previous_tab_name != MainTabNames.SIMULATION_RUNNER.value
        ):
            # Entered SIMULATION_RUNNER tab
            self.simulation_runner_tab.init_simulations()

        if (
                self.previous_tab_name == MainTabNames.SIMULATION_RUNNER.value
                and new_tab_name != MainTabNames.SIMULATION_RUNNER.value
        ):
            # Left SIMULATION_RUNNER tab
            self.simulation_runner_tab.reset()

        if (
                new_tab_name == MainTabNames.VISUALIZER.value
                and self.previous_tab_name != MainTabNames.VISUALIZER.value
        ):
            self.visualizer_tab.init_visualization()

        if (
                self.previous_tab_name == MainTabNames.VISUALIZER.value
                and new_tab_name != MainTabNames.VISUALIZER.value
        ):
            self.visualizer_tab.reset()

        self.previous_tab_name = new_tab_name


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainAppWindow()
    main_window.show()
    sys.exit(app.exec_())