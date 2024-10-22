import sys

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QApplication,
    QAbstractItemView,
    QLabel,
    QSizePolicy,
)

from application.application_status import ApplicationStatus
from control.mpc.weights import Weights, InputWeights, OutputWeights


class WeightsTableWidget(QWidget):
    weightsChanged = pyqtSignal(Weights, name="weightsChanged")

    def __init__(self, current_app_status: ApplicationStatus, parent=None):
        super().__init__(parent=parent)
        self.current_app_status = current_app_status
        self.ignore_itm_change = False
        self.current_weights = None

        self.layout = QVBoxLayout(self)
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            [
                "Velocity",
                "Heading Angle",
                "X Global",
                "Y Global",
                "Acceleration",
                "Steering Angle",
            ]
        )
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setStyleSheet(
            """
            QTableWidget {
                background-color: """
            + self.current_app_status.color_theme.background_color
            + """;
                selection-background-color: red;
                border: 1px solid #000000;
            }

            QHeaderView::section {
                background-color: """
            + self.current_app_status.color_theme.secondary_color
            + """;
                color: """
            + self.current_app_status.color_theme.text_color
            + """;
                padding: 5px;
                border: none;
            }

            QTableCornerButton::section {
                background-color: """
            + self.current_app_status.color_theme.secondary_color
            + """;
                border: none;
            }
            
            QTableWidget::item {
                color: """
            + self.current_app_status.color_theme.text_color
            + """;
            }
            
            QTableWidget::item:selected {
                color: """
            + self.current_app_status.color_theme.text_color
            + """;
            }
        """
        )

        self.populateTable(define_weights())

        text_label = QLabel("Weights:")
        text_label.setStyleSheet(
            f"color: {self.current_app_status.color_theme.text_color}; font-size: 16px;"
        )
        self.layout.addWidget(text_label)
        self.layout.addWidget(self.table, stretch=2)
        self.setLayout(self.layout)

        self.table.itemSelectionChanged.connect(self.handle_selection_change)
        self.table.itemChanged.connect(self.handle_item_changed)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.table.setCurrentCell(0, 0)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def populateTable(self, weights):
        self.table.setRowCount(len(weights))
        for i, weight in enumerate(weights):
            for j, val in enumerate(weight):
                self.table.setItem(i, j, QTableWidgetItem(str(val)))

        # Add an empty row at the end
        self.add_row()

    def add_row(self):
        row_position = self.table.rowCount()
        self.table.insertRow(row_position)

    def handle_selection_change(self):
        if self.ignore_itm_change:
            return
        selected_row = self.table.currentRow()
        if selected_row != self.table.rowCount() - 1:
            self.ignore_itm_change = True
            for i in range(self.table.rowCount()):
                for j in range(self.table.columnCount()):
                    item = self.table.item(i, j)
                    if item is not None:
                        item.setBackground(
                            QColor(self.current_app_status.color_theme.secondary_color)
                        )
            for j in range(self.table.columnCount()):
                item = self.table.item(selected_row, j)
                if item is not None:
                    item.setBackground(
                        QColor(self.current_app_status.color_theme.primary_color)
                    )

            self.ignore_itm_change = False
            self.create_weights_object(selected_row)

    def handle_item_changed(self, item):
        if self.ignore_itm_change:
            return
        selected_row = self.table.currentRow()
        if selected_row == self.table.rowCount() - 1:
            self.populate_default_values(item.row())
            self.add_row()
        self.create_weights_object(selected_row)

    def create_weights_object(self, row):
        Q_weights = OutputWeights(
            velocity=float(self.table.item(row, 0).text()),
            heading_angle=float(self.table.item(row, 1).text()),
            X_global=float(self.table.item(row, 2).text()),
            Y_global=float(self.table.item(row, 3).text()),
        )
        S_weights = Q_weights
        R_weights = InputWeights(
            acceleration=float(self.table.item(row, 4).text()),
            steering_angle=float(self.table.item(row, 5).text()),
        )
        weights_object = Weights(Q_weights, S_weights, R_weights)
        print("Weights object created:")
        for key, value in weights_object.__dict__.items():
            print(f"{key}: {value}")
        self.current_weights = weights_object
        self.weightsChanged.emit(weights_object)

    def populate_default_values(self, row):
        self.ignore_itm_change = True
        for j in range(self.table.columnCount()):
            item = self.table.item(row, j)
            if item is None or item.text() == "":
                default_value = 0.0
                item = QTableWidgetItem(str(default_value))
                item.setFlags(item.flags() | Qt.ItemIsEditable)
                self.table.setItem(row, j, item)
        self.ignore_itm_change = False


def define_weights():
    weights = [
        (1.0, 200.0, 50.0, 50.0, 1.0, 100.0),
        (1.0, 200.0, 50.0, 50.0, 1.0, 100.0),
        (100.0, 20000.0, 1000.0, 1000.0, 100.0, 1.0),
        (100.0, 20000.0, 1000.0, 1000.0, 100.0, 1.0),
        (1.0, 200.0, 10000.0, 10000.0, 1.0, 100.0),
    ]
    return weights


if __name__ == "__main__":
    app = QApplication(sys.argv)
    weight_widget = WeightsTableWidget()
    weight_widget.show()
    sys.exit(app.exec_())

# def define_weights():
#     Q_weights = OutputWeights(
#         velocity=1.0, heading_angle=200.0, X_global=50.0, Y_global=50.0
#     )
#     S_weights = Q_weights
#     R_weights = InputWeights(acceleration=1.0, steering_angle=100.0)
#     w1 = Weights(Q_weights, S_weights, R_weights)
#     w2 = w1
#     Q_weights = OutputWeights(
#         velocity=100.0, heading_angle=20000.0, X_global=1000.0, Y_global=1000.0
#     )
#     S_weights = Q_weights
#     R_weights = InputWeights(acceleration=100.0, steering_angle=1)
#     w3 = Weights(Q_weights, S_weights, R_weights)
#     w4 = w3
#     Q_weights = OutputWeights(
#         velocity=1.0, heading_angle=200.0, X_global=10000.0, Y_global=10000.0
#     )
#     S_weights = OutputWeights(
#         velocity=5.0, heading_angle=300.0, X_global=50000.0, Y_global=50000.0
#     )
#     R_weights = InputWeights(acceleration=1.0, steering_angle=100.0)
#     w5 = Weights(Q_weights, S_weights, R_weights)
#
#     return w1, w2, w3, w4, w5