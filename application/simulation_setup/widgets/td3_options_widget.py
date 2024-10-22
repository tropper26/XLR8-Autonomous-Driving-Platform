from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout

from application.application_status import ApplicationStatus
from application.core.generic_form_widget import GenericFormWidget
from simulation.simulation_info import SimulationInfo


class TD3OptionsWidget(QWidget):
    def __init__(
        self,
        parent,
        current_app_status: ApplicationStatus,
        current_simulation_info: SimulationInfo,
    ):
        super().__init__(parent)

        self.current_app_status = current_app_status
        self.current_simulation_info = current_simulation_info

        self.layout = QVBoxLayout(self)

        self.horizontal_widget = QWidget(parent=self)
        self.horizontal_layout = QHBoxLayout(self.horizontal_widget)
        self.layout.addWidget(self.horizontal_widget)

        self.controller_params = self.current_simulation_info.get_controller_params(
            self.current_simulation_info.controller_name,
            self.current_simulation_info.controller_params_name,
            self.current_app_status,
        )
        print("TD3OptionsWidget: controller_params:", self.controller_params)
        self.params_form = GenericFormWidget(
            self.controller_params,
            "TD3 Parameters",
            color_theme=self.current_app_status.color_theme,
        )
        self.horizontal_layout.addWidget(
            self.params_form, alignment=Qt.AlignTop, stretch=1
        )

        self.layout.addStretch(1)