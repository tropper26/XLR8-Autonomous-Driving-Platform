from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QScrollArea,
    QFrame,
)

from application.application_status import ApplicationStatus
from application.simulation_runner.widgets.simulation_widget import SimulationWidget
from simulation.simulation_info import SimulationResult


class SimulationRunnerTab(QWidget):
    def __init__(self, parent, current_app_status: ApplicationStatus):
        super().__init__(parent=parent)
        self.vertical_layout = None
        self.vertical_widget = None
        self.scroll_area = None
        self.current_app_status = current_app_status
        self.layout = QVBoxLayout(self)

        self.init_ui()

    def init_ui(self):
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )  # Options: Qt.ScrollBarAlwaysOff, Qt.ScrollBarAsNeeded, Qt.ScrollBarAlwaysOn

        self.vertical_widget = QWidget(self.scroll_area)
        self.vertical_widget.setStyleSheet(
            f"background-color: {self.current_app_status.color_theme.background_color};"
        )
        self.vertical_layout = QVBoxLayout(self.vertical_widget)

        self.scroll_area.setWidget(self.vertical_widget)

        self.layout.addWidget(self.scroll_area)

    def reset(self):
        # remove all widgets from the layout
        for i in reversed(range(self.vertical_layout.count())):
            widget = self.vertical_layout.itemAt(i).widget()
            widget.deleteLater()

    def update_simulation_results(self, sim_result: SimulationResult):
        if sim_result is None:
            raise RuntimeError("Simulation result is None")

        self.current_app_status.simulation_results[
            sim_result.simulation_info.identifier
        ] = sim_result

    def init_simulations(self):
        delimiter = QFrame(self.vertical_widget)
        delimiter.setFrameShape(QFrame.HLine)  # Set the shape to a horizontal line
        delimiter.setFrameShadow(QFrame.Sunken)  # Gives a sunken effect to the line

        simulation_infos = self.current_app_status.simulation_infos
        for index, simulation_info in enumerate(simulation_infos):
            if index > 0:  # Add a delimiter between the simulation widgets
                self.vertical_layout.addWidget(delimiter)

            simulation_widget = SimulationWidget(
                self, simulation_info, self.current_app_status
            )
            simulation_widget.selectedResultChanged.connect(
                self.update_simulation_results
            )
            simulation_widget.setStyleSheet(
                f"""
                background-color: {self.current_app_status.color_theme.secondary_color};
                border: 2px solid black;
                """
            )
            simulation_widget.setFixedHeight(1200)
            self.vertical_layout.addWidget(simulation_widget)