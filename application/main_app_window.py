import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
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


class MainAppWindow(QMainWindow):
    def __init__(self, current_app_status: ApplicationStatus):
        super().__init__()
        self.simulation_setup_widget = None
        self.setWindowTitle("Autonomous Vehicle Simulator")
        self.resize(1700, 850)
        self.current_app_status = current_app_status
        self.previous_tab_name = None

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

        self.tab_widget = QTabWidget()
        self.layout.addWidget(self.tab_widget)

        self.global_planner_tab = GlobalPlannerTab(self, self.current_app_status)
        self.tab_widget.addTab(self.global_planner_tab, "Global Planning")

        self.simulations_tabs = DynamicTabWidget(
            tabs_factory=self.add_simulation,
            tabs_name="Simulation",
            color_theme=self.current_app_status.color_theme,
            parent=self,
        )
        self.tab_widget.addTab(self.simulations_tabs, "Setup Simulations")

        self.simulation_runner_tab = SimulationRunnerTab(self, self.current_app_status)
        self.tab_widget.addTab(self.simulation_runner_tab, "Run Simulations")

        self.visualizer_tab = VisualizerTab(self, self.current_app_status)
        self.tab_widget.addTab(self.visualizer_tab, "Visualize Simulations")

        # Connect the tabChanged signal to the custom method tab_changed
        self.tab_widget.tabBarClicked.connect(self.tab_changed)

        self.fullscreen = False

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

    def tab_changed(self, index):
        tab_name = self.tab_widget.tabText(index)
        print(f"Tab changed to: {tab_name}")

        if tab_name == "Run Simulations":
            self.simulation_runner_tab.init_simulations()

        if self.previous_tab_name == "Run Simulations":
            self.simulation_runner_tab.reset()

        if tab_name == "Visualize Simulations":
            self.visualizer_tab.init_visualization()

        if self.previous_tab_name == "Visualize Simulations":
            self.visualizer_tab.reset()

        self.previous_tab_name = tab_name


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainAppWindow()
    main_window.show()
    sys.exit(app.exec_())