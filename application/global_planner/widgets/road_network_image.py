import os
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QWidget, QVBoxLayout
from PyQt5.QtGui import QPixmap
from dto.color_theme import ColorTheme


class RoadNetworkImage(QWidget):
    def __init__(self, road_network_name: str, path_to_image: str, path_to_no_figure_found: str,
                 color_theme: ColorTheme, parent=None):
        super(RoadNetworkImage, self).__init__(parent)

        self.road_network_image = road_network_name
        self.path_to_image = os.path.abspath(path_to_image)
        self.path_to_no_figure_found = os.path.abspath(path_to_no_figure_found)
        self.color_theme = color_theme

        self.layout = QVBoxLayout()

        self.text_label = QLabel(self.road_network_image, self)
        self.text_label.setAlignment(Qt.AlignCenter)
        self.text_label.setStyleSheet(
            f"color: {color_theme.button_text_color};"
            f"font-size: 20px;"
        )
        self.layout.addWidget(self.text_label)

        self.pixmap_label = QLabel(self)

        pixmap = QPixmap(self.path_to_image)
        if pixmap.isNull():
            print(f"Failed to load image: {self.path_to_image}, loading fallback image.")
            pixmap = QPixmap(self.path_to_no_figure_found)

        self.pixmap_label.setPixmap(pixmap)
        self.pixmap_label.setStyleSheet(
            f"border: 8px;"
            f"border-style: solid;"
            f"border-radius: 10px;"
            f"border-color: {self.color_theme.secondary_color};"
        )

        self.layout.addWidget(self.pixmap_label)

        self.setLayout(self.layout)

    def setFixedSize(self, width: int, height: int):
        current_pixmap = self.pixmap_label.pixmap()
        if current_pixmap is not None:
            resized_pixmap = current_pixmap.scaled(width, height, Qt.KeepAspectRatio)

            self.pixmap_label.setPixmap(resized_pixmap)

            self.pixmap_label.resize(resized_pixmap.width(), resized_pixmap.height())

            self.resize(resized_pixmap.width(), resized_pixmap.height() + self.text_label.height())
