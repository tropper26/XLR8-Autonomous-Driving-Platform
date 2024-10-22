from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
)

from application.application_status import ApplicationStatus
from application.core.generic_button_group_widget import GenericButtonGroupWidget
from application.core.option_selector_widget import OptionSelectorWidget
from application.simulation_setup.widgets.controller_selection_widget import (
    ControllerSelectionWidget,
)


class SimulationSetupTab(QWidget):
    def __init__(
        self, parent, current_app_status: ApplicationStatus, simulation_index: int
    ):
        super().__init__(parent=parent)

        self.current_app_status = current_app_status
        self.current_simulation_info = self.current_app_status.simulation_infos[
            simulation_index - 1
        ]
        self.setGeometry(parent.geometry())
        self.setStyleSheet(
            f"background-color: {self.current_app_status.color_theme.background_color};"
        )

        self.layout = QHBoxLayout(self)

        self.left_widget = QWidget()
        self.left_layout = QVBoxLayout(self.left_widget)
        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout(self.right_widget)

        self.vehicle_params_widget = OptionSelectorWidget(
            color_theme=self.current_app_status.color_theme,
            options_dict=self.current_app_status.vehicle_params_lookup,
            initial_selected_key=self.current_simulation_info.vehicle_params_name,
            title="Vehicle Params",
        )
        self.vehicle_params_widget.optionSelected.connect(self.set_vehicle_params)

        self.constraints_widget = OptionSelectorWidget(
            color_theme=self.current_app_status.color_theme,
            options_dict=self.current_app_status.static_constraints_lookup,
            initial_selected_key=self.current_simulation_info.static_constraints_name,
            title="Constraints",
        )
        self.constraints_widget.optionSelected.connect(self.set_static_constraints)

        self.controller_widget = ControllerSelectionWidget(
            self, self.current_app_status, simulation_index
        )

        # self.right_layout.addStretch(1)

        self.vehicle_model_widget = QWidget()
        self.vehicle_model_layout = QVBoxLayout(self.vehicle_model_widget)

        self.vehicle_model_button_group = GenericButtonGroupWidget(
            options=self.current_app_status.vehicle_model_options,
            initial_selection=self.current_simulation_info.vehicle_model_name,
            active_button_style=f"background-color: {self.current_app_status.color_theme.selected_color}; "
            f"color: {self.current_app_status.color_theme.button_text_color};",
            inactive_button_style=f"background-color: {self.current_app_status.color_theme.secondary_color}; "
            f"color: {self.current_app_status.color_theme.button_text_color};",
        )

        self.vehicle_model_button_group.optionsSelected.connect(self.set_vehicle_model)

        self.vehicle_model_layout.addWidget(
            self.vehicle_model_button_group, alignment=Qt.AlignTop
        )

        self.layout.addWidget(self.left_widget, stretch=1)
        self.layout.addWidget(self.right_widget, stretch=2)
        self.left_layout.addWidget(self.vehicle_params_widget)
        self.left_layout.addWidget(self.constraints_widget)
        self.right_layout.addWidget(self.vehicle_model_widget)
        self.right_layout.addWidget(self.controller_widget)

    def set_vehicle_model(self, vehicle_model_names):
        vehicle_model_name = vehicle_model_names[
            0
        ]  # button group only allows single selection
        self.current_simulation_info.vehicle_model_name = vehicle_model_name

    def set_vehicle_params(self, vehicle_params_name):
        self.current_simulation_info.vehicle_params_name = vehicle_params_name

    def set_static_constraints(self, static_constraints_name):
        self.current_simulation_info.static_constraints_name = static_constraints_name