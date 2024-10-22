from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QCheckBox, QHBoxLayout

from application.application_status import ApplicationStatus
from application.core.generic_form_widget import GenericFormWidget
from control.reinforcement_learning.tuning.auto_tuner_params import AutoTunerParams
from simulation.simulation_info import SimulationInfo


class PurePursuitOptionsWidget(QWidget):
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
        self.params_form = GenericFormWidget(
            self.controller_params,
            "Pure Pursuit Parameters",
            color_theme=self.current_app_status.color_theme,
        )
        self.horizontal_layout.addWidget(
            self.params_form, alignment=Qt.AlignTop, stretch=1
        )

        self.tuning_params_widget = QWidget(parent=self.horizontal_widget)
        self.tuning_params_layout = QVBoxLayout(self.tuning_params_widget)
        self.horizontal_layout.addWidget(self.tuning_params_widget, stretch=1)

        self.checkbox = QCheckBox("Tune Parameters")
        self.checkbox.stateChanged.connect(self.toggle_tuneable_params)
        self.tuning_params_layout.addWidget(self.checkbox, alignment=Qt.AlignHCenter)

        self.auto_tuner_params = AutoTunerParams()
        self.tuneable_params_form = GenericFormWidget(
            self.auto_tuner_params,
            "Training Parameters for Hyperparameter Tuning",
            color_theme=self.current_app_status.color_theme,
        )
        self.tuneable_params_form.hide()
        self.tuning_params_layout.addWidget(
            self.tuneable_params_form, alignment=Qt.AlignTop
        )
        self.tuning_params_layout.addStretch(1)

        self.layout.addStretch(1)

    def toggle_tuneable_params(self, state):
        if state == Qt.Checked:
            self.tuneable_params_form.show()
            self.controller_params.tuning_params = self.auto_tuner_params
        else:
            self.tuneable_params_form.hide()
            self.controller_params.tuning_params = None