from PyQt5.QtWidgets import QScrollArea, QWidget, QVBoxLayout, QHBoxLayout


class ScrollAbleWidgetList(QWidget):
    def __init__(self, widget_list, widgets_per_row):
        super().__init__()

        self.widget_list = widget_list
        self.widgets_per_row = widgets_per_row

        self.init_ui()

    def init_ui(self):
        # Create the scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        # Create a container widget for the scroll area
        container = QVBoxLayout()
        scroll_area.setWidget(container)

        # Create a vertical layout for the container widget
        layout = QVBoxLayout(container)

        # Populate the layout with widgets
        for i, widget in enumerate(self.widget_list):
            if i % self.widgets_per_row == 0:
                # Create a new horizontal layout for each row
                row_layout = QHBoxLayout()
                layout.addLayout(row_layout, stretch=1)
            # Add widget to the current row layout
            row_layout.addWidget(widget)

        self.setLayout(layout)