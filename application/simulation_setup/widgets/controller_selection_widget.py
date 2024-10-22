from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout

from application.application_status import ApplicationStatus
from application.core.generic_button_group_widget import GenericButtonGroupWidget
from application.simulation_setup.widgets.mpc_options_widget import MPCOptionsWidget
from application.simulation_setup.widgets.pure_pursuit_options_widget import (
    PurePursuitOptionsWidget,
)
from application.simulation_setup.widgets.sac_options_widget import SACOptionsWidget
from application.simulation_setup.widgets.stanley_options_widget import (
    StanleyOptionsWidget,
)
from application.simulation_setup.widgets.td3_options_widget import TD3OptionsWidget


class ControllerSelectionWidget(QWidget):
    def __init__(
        self, parent, current_app_status: ApplicationStatus, simulation_index: int
    ):
        super().__init__(parent)
        self.current_app_status = current_app_status
        self.current_simulation_info = self.current_app_status.simulation_infos[
            simulation_index - 1
        ]
        self.layout = QVBoxLayout(self)

        self.controller_button_group = GenericButtonGroupWidget(
            options=self.current_app_status.controller_options,
            initial_selection=self.current_simulation_info.controller_name,
            active_button_style=f"background-color: {self.current_app_status.color_theme.selected_color}; "
            f"color: {self.current_app_status.color_theme.button_text_color};",
            inactive_button_style=f"background-color: {self.current_app_status.color_theme.secondary_color};"
            f"color: {self.current_app_status.color_theme.button_text_color};",
        )
        self.controller_button_group.optionsSelected.connect(
            self.update_controller_options
        )

        self.controller_options = self.get_controller_options_widget(
            self.current_simulation_info.controller_name,
        )

        self.layout.addWidget(self.controller_button_group, alignment=Qt.AlignTop)
        self.layout.addWidget(self.controller_options, alignment=Qt.AlignTop)
        self.layout.addStretch(1)

    def get_controller_options_widget(self, controller_name):
        if controller_name == "Model Predictive Controller":
            return MPCOptionsWidget(
                self, self.current_app_status, self.current_simulation_info
            )
        elif controller_name == "Pure Pursuit Controller":
            return PurePursuitOptionsWidget(
                self, self.current_app_status, self.current_simulation_info
            )
        elif controller_name == "Stanley Controller":
            return StanleyOptionsWidget(
                self, self.current_app_status, self.current_simulation_info
            )
        elif controller_name == "Soft Actor Critic":
            return SACOptionsWidget(
                self, self.current_app_status, self.current_simulation_info
            )
        elif controller_name == "TD3":
            return TD3OptionsWidget(
                self, self.current_app_status, self.current_simulation_info
            )
        else:
            return QWidget()  # Return a placeholder widget if controller not recognized

    def update_controller_options(self, controller_names):
        controller_name = controller_names[0]  # button group is single select
        self.current_simulation_info.controller_name = controller_name

        self.layout.removeWidget(self.controller_options)
        self.controller_options.deleteLater()

        self.controller_options = self.get_controller_options_widget(controller_name)
        self.layout.addWidget(self.controller_options, alignment=Qt.AlignTop)