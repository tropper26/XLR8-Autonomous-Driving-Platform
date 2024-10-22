import numpy as np
from PyQt5.QtWidgets import QApplication

from application.application_status import ApplicationStatus
from application.main_app_window import MainAppWindow

if __name__ == "__main__":
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    np.set_printoptions(suppress=True, linewidth=9999999999)
    application = QApplication([])

    current_app_status = ApplicationStatus()

    global_stylesheet = (
        """
        QMainWindow {
            background-color: #333;
            color: #FFF;
        }
        QLabel {
            font-weight: bold;
        }
        QPushButton {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            font-size: 14px;
            font-weight: bold;
            color: """
        + current_app_status.color_theme.button_text_color
        + """;
            background-color: """
        + current_app_status.color_theme.secondary_color
        + """;
        }
        QPushButton:hover {
            background-color: #888;
        }
        QTabWidget::pane {
                border: 1px solid """
        + current_app_status.color_theme.primary_color
        + """;
                background-color: """
        + current_app_status.color_theme.primary_color
        + """;
            }

            QTabBar::tab {
                background-color: """
        + current_app_status.color_theme.secondary_color
        + """;
                color: """
        + current_app_status.color_theme.button_text_color
        + """;
                padding: 18px 32px; 
                border-top-left-radius: 7px;
                border-top-right-radius: 7px;
                font-size: 16px; 
                font-weight: bold;
                min-width: 200px; 
                min-height: 20px;
            }

            QTabBar::tab:selected {
                background-color:  """
        + "red"
        + """;
            }
    """
    )

    application.setStyleSheet(global_stylesheet)

    main_app_window = MainAppWindow(current_app_status)

    main_app_window.show()
    application.exec_()