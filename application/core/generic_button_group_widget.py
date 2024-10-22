from PyQt5 import QtCore
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QButtonGroup


class GenericButtonGroupWidget(QWidget):
    optionsSelected = QtCore.pyqtSignal(
        list, name="optionsSelected"
    )  # Define a signal to emit the selected option

    def __init__(
        self,
        options: list[str],
        initial_selection: (
            str,
            list[str],
        ) = None,  # Change the type hint to Union[str, list[str]] to allow for multi-select
        disabled_options=None,
        vertical_layout=False,
        multi_select=False,
        default_button_style: str = None,
        active_button_style: str = None,
        inactive_button_style: str = None,
        disabled_button_style: str = None,
        parent=None,
    ):
        super().__init__(parent)
        self.options = options
        self.vertical_layout = vertical_layout
        self.multi_select = multi_select  # Store the multi-select flag

        if initial_selection is not None:
            if self.multi_select and not isinstance(initial_selection, list):
                raise ValueError(
                    "Initial selection must be a list when multi-select is enabled"
                )

            if not self.multi_select and isinstance(initial_selection, list):
                raise ValueError(
                    "Initial selection must be a single value when multi-select is disabled"
                )

        self.styles_setup(
            default_button_style,
            active_button_style,
            inactive_button_style,
            disabled_button_style,
        )

        self.init_ui(initial_selection, disabled_options)

    def init_ui(self, initial_selection, disabled_options):
        self.layout = QVBoxLayout() if self.vertical_layout else QHBoxLayout()
        self.setLayout(self.layout)
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(
            not self.multi_select
        )  # Set exclusivity based on multi-select

        for index, option in enumerate(self.options):
            button = QPushButton(option)
            button.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
            button.setCheckable(True)  # Make buttons checkable for multi-select
            if disabled_options is not None and option in disabled_options:
                button.setEnabled(False)
                button.setStyleSheet(self.disabled_button_style)
            self.button_group.addButton(button, index)
            self.layout.addWidget(button)

        # Check the initial selections
        if initial_selection is not None:
            for button in self.button_group.buttons():
                if button.text() in initial_selection and button.isEnabled():
                    button.setChecked(True)

        # Call the options_selected method to emit the initial selections
        self.options_selected()

        # Connect the clicked signal of each button to the option_selected method
        self.button_group.buttonClicked.connect(self.options_selected)

    def update_button_styles(self):
        for button in self.button_group.buttons():
            if button.isEnabled():
                if button.isChecked():
                    button.setStyleSheet(self.active_button_style)
                else:
                    button.setStyleSheet(self.inactive_button_style)

    def options_selected(self):
        self.update_button_styles()
        selected_options = [
            button.text()
            for button in self.button_group.buttons()
            if button.isChecked()
        ]
        print("Selected options:", selected_options)
        self.optionsSelected.emit(selected_options)

    def styles_setup(
        self,
        default_button_style,
        active_button_style,
        inactive_button_style,
        disabled_button_style,
    ):
        if default_button_style is None:
            default_button_style = """
                        padding: 10px 20px;
                        border: 2px solid black;
                        border-radius: 5px;
                        font-size: 14px;
                        """
        if active_button_style is None:
            active_button_style = """
                color: #fff;
                background-color: #28a745;
                """
        self.active_button_style = default_button_style + active_button_style
        if inactive_button_style is None:
            inactive_button_style = """
                color: #fff;
                background-color: #007bff;
                """
        self.inactive_button_style = default_button_style + inactive_button_style

        if disabled_button_style is None:
            disabled_button_style = """
                color: #fff;
                background-color: #6c757d;
                """

        self.disabled_button_style = default_button_style + disabled_button_style