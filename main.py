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

    application.setStyleSheet(current_app_status.global_stylesheet)

    main_app_window = MainAppWindow(current_app_status)

    main_app_window.show()
    application.exec_()