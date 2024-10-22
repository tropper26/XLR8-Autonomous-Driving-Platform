from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
)

from application.core.generic_button_group_widget import GenericButtonGroupWidget
from application.core.generic_form_widget import GenericFormWidget
from dto.color_theme import ColorTheme


class OptionSelectorWidget(QWidget):
    optionSelected = pyqtSignal(object)

    def __init__(
        self,
        color_theme: ColorTheme,
        options_dict: dict[str, object],
        initial_selected_key: str,
        title: str,
        readonly=True,
        parent=None,
    ):
        super().__init__(parent=parent)
        self.options_dict = options_dict
        self.layout = QHBoxLayout(self)

        self.left_widget = QWidget()
        self.left_layout = QVBoxLayout(self.left_widget)
        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout(self.right_widget)
        self.layout.addWidget(self.left_widget, stretch=1)
        self.layout.addWidget(self.right_widget, stretch=2)

        # Add radio buttons for each item in the dictionary
        self.generic_button_group = GenericButtonGroupWidget(
            options=list(options_dict.keys()),
            initial_selection=initial_selected_key,
            vertical_layout=True,
            active_button_style=f"background-color: {color_theme.selected_color}; color: {color_theme.button_text_color};",
            inactive_button_style=f"background-color: {color_theme.secondary_color}; color: {color_theme.button_text_color};",
        )

        self.generic_button_group.optionsSelected.connect(self.keys_chosen)

        print("initial_selected_key", initial_selected_key)

        self.generic_form_widget = GenericFormWidget(
            instance=self.options_dict[initial_selected_key],
            title=initial_selected_key,
            color_theme=color_theme,
            editable=readonly,
        )
        qlabel = QLabel(f"Select {title}")
        qlabel.setStyleSheet(f"color: {color_theme.text_color}; font-size: 16px;")
        self.left_layout.addWidget(qlabel, alignment=Qt.AlignCenter)
        self.left_layout.addWidget(self.generic_button_group)
        self.left_layout.addStretch(1)
        self.right_layout.addWidget(self.generic_form_widget)

    def keys_chosen(self, keys: list[str]):
        key = keys[0]  # Button group is exclusive, so only one key will be selected
        self.generic_form_widget.update_instance(self.options_dict[key], key)

        self.optionSelected.emit(key)