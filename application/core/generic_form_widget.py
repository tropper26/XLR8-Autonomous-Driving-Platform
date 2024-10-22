from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFormLayout,
)

from dto.color_theme import ColorTheme
from dto.form_dto import FormDTO


class GenericFormWidget(QWidget):
    formChanged = pyqtSignal(object, name="formChanged")

    def __init__(
        self,
        instance: FormDTO,
        title: str,
        color_theme: ColorTheme,
        editable=True,
        parent=None,
    ):
        super().__init__(parent=parent)
        self.instance = instance
        self.title = title
        self.color_theme = color_theme
        self.editable = editable
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.form_layout = QFormLayout()
        layout.addLayout(self.form_layout)

        # Display class title and instance variable name on the first row
        self.class_title_label = QLabel(self.title)
        self.class_title_label.setStyleSheet(
            f"color: {self.color_theme.text_color}; font-size: 16px;"
        )
        self.form_layout.addRow(self.class_title_label)

        self.line_edits = {}  # Store line edits in a dictionary
        attributes_to_ignore = self.instance.attributes_to_ignore()

        for attr, value in vars(self.instance).items():
            if attr in attributes_to_ignore:
                continue
            label = QLabel(attr.capitalize().replace("_", " "))
            label.setStyleSheet(
                f"color: {self.color_theme.text_color}; font-size: 14px;"
            )

            line_edit = QLineEdit(str(value))
            line_edit.setStyleSheet(
                f"background-color: {self.color_theme.secondary_color}; color: {self.color_theme.text_color};"
            )
            line_edit.setReadOnly(
                not self.editable
            )  # Set line edit to read-only if not editable

            # line_edit.setMaximumWidth(100)

            self.form_layout.addRow(label, line_edit)
            self.line_edits[attr] = line_edit  # Store line edit reference

        if self.editable:
            save_button = QPushButton("Save")
            save_button.clicked.connect(self.save_values)
            save_button.setStyleSheet(
                """
                    margin-top: 20px;    
                    padding: 10px 20px;
                    border: 2px solid black;
                    border-radius: 5px;
                """
                + f"background-color: {self.color_theme.selected_color}; color: {self.color_theme.button_text_color};"
            )
            layout.addWidget(save_button)
        layout.addStretch()
        self.setLayout(layout)

    def update_instance(
        self, instance: FormDTO, title
    ):  # Update the form with a new instance
        if not isinstance(instance, type(self.instance)):
            raise ValueError(
                f"Instance is not of the same class. Expected {type(self.instance)}, got {type(instance)}"
            )
        self.instance = instance
        self.title = title
        self.class_title_label.setText(title)
        for attr, value in vars(self.instance).items():
            if attr in self.line_edits:  # Update line edits with new values
                self.line_edits[attr].setText(str(value))

    def save_values(self):  # Save the values from the line edits to the instance
        for attr in vars(self.instance):
            print(attr)
            line_edit = self.line_edits.get(attr)
            if line_edit is not None:
                setattr(self.instance, attr, float(line_edit.text()))
        self.formChanged.emit(self.instance)