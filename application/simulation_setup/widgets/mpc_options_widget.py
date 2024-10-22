from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout

from application.simulation_setup.widgets.weights_table_widget import WeightsTableWidget
from control.mpc.mpc_params import MPCParams
from application.application_status import ApplicationStatus
from application.core.generic_form_widget import GenericFormWidget


class MPCOptionsWidget(QWidget):
    def __init__(
        self, parent, current_app_status: ApplicationStatus, current_simulation_info
    ):
        super().__init__(parent)

        self.current_app_status = current_app_status
        self.current_simulation_info = current_simulation_info

        self.layout = QVBoxLayout(self)

        self.horizontal_widget = QWidget(parent=self)
        self.horizontal_layout = QHBoxLayout(self.horizontal_widget)

        self.controller_params = self.current_simulation_info.get_controller_params(
            self.current_simulation_info.controller_name,
            self.current_simulation_info.controller_params_name,
            self.current_app_status,
        )
        self.params_form = GenericFormWidget(
            self.controller_params,
            "MPC Parameters",
            self.current_app_status.color_theme,
        )
        self.stretch_widget = QWidget(parent=self.horizontal_widget)

        self.weights_table = WeightsTableWidget(
            current_app_status=self.current_app_status, parent=self
        )

        self.layout.addWidget(self.horizontal_widget, alignment=Qt.AlignTop)
        self.horizontal_layout.addWidget(
            self.params_form, alignment=Qt.AlignTop, stretch=1
        )
        self.horizontal_layout.addWidget(self.stretch_widget, stretch=3)
        self.layout.addWidget(self.weights_table, alignment=Qt.AlignTop)