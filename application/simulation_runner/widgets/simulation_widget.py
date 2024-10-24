import multiprocessing as mp
import time

import numpy as np
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
)

from application.application_status import ApplicationStatus
from application.core.generic_button_group_widget import GenericButtonGroupWidget
from application.core.generic_form_widget import GenericFormWidget
from application.visualizer.widgets.simulation_visualizer_widget import (
    SimulationVisualizerWidget,
)
from dto.color_theme import ColorTheme
from simulation.simulation_process import SimulationProcess
from simulation.simulation_info import SimulationInfo
from simulation.simulation_result import SimulationResult


class SimulationWidget(QWidget):
    selectedResultChanged = pyqtSignal(object, name="selectedResultChanged")

    def __init__(
        self,
        parent,
        simulation_info: SimulationInfo,
        current_app_status: ApplicationStatus,
    ):
        super().__init__(parent=parent)
        self.sim_process = None
        self.result_queue = None
        self.layout = QHBoxLayout(self)
        self.simulation_info = simulation_info
        self.current_app_status = current_app_status
        self.results: list[SimulationResult] = []
        self.current_sim_result_index = 0

        # Shared multiprocessing Values
        self.sim_run_count = mp.Value("i", 1)
        self.visualization_interval = mp.Value("i", 1)

        self.errors = self.simulation_info.validate()
        self.errors.extend(self.current_app_status.validate())
        self.init_ui()

        if not self.errors:
            self.start_simulation()

    def deleteLater(self):
        self.stop_simulation()
        super().deleteLater()

    def start_simulation(self):
        start_time = time.perf_counter()
        self.sim_visualizer_widget.setup_ref_path(
            self.current_app_status.ref_path.discretized,
            self.current_app_status.path_obstacles,
            self.current_app_status.world_x_limit,
            self.current_app_status.world_y_limit,
        )

        self.result_queue = mp.Queue()
        self.params_pipe, child_pipe = mp.Pipe()
        self.sim_process = SimulationProcess(
            simulation_info=self.simulation_info,
            app_status=self.current_app_status,
            result_queue=self.result_queue,
            controller_params_pipe=child_pipe,
            sim_run_count=self.sim_run_count,
            visualization_interval=self.visualization_interval,
        )
        print(f"Before Started simulation in {time.perf_counter() - start_time:.2f} s")
        self.sim_process.start()
        print(f"Started simulation in {time.perf_counter() - start_time:.2f} s")
        self.monitor_simulation()


    def stop_simulation(self):
        if self.sim_process and self.sim_process.is_alive():
            self.sim_process.terminate()
            print("Simulation process stopped.")

        if self.sim_visualizer_widget:
            self.sim_visualizer_widget.deleteLater()

    def monitor_simulation(self):
        while not self.result_queue.empty():
            # start_time = time.perf_counter()
            # print(f"got new result at {start_time}")
            result = self.result_queue.get()
            if isinstance(result, SimulationResult):
                print(f"Received  result: {result.run_index}")
                if not self.results:  # First result
                    self.sim_visualizer_widget.visualize_results([result.copy()])
                    self.update_run_info(
                        result.iteration_info_batch.time,
                        result.iteration_info_batch.execution_time,
                    )
                    self.current_sim_result_index = 0
                    self.selectedResultChanged.emit(result.copy())

                self.results.append(result.copy())
                self.run_label.setText(
                    f"Visualizing Run {self.current_sim_result_index+1}:{len(self.results)} Total Runs"
                )
            # print(f"Finished processing result at {time.perf_counter() - start_time:.2f} s")

        QTimer.singleShot(100, self.monitor_simulation)

    def init_top_bar(self):
        self.top_bar = QWidget(self)
        self.top_layout = QHBoxLayout(self.top_bar)
        self.left_layout.addWidget(self.top_bar)

        options = ["<<<", "<<", "<", ">", ">>", ">>>"]

        self.episode_button_group = GenericButtonGroupWidget(
            options=options,
            multi_select=False,
            active_button_style=f"background-color: {self.current_app_status.color_theme.selected_color}; "
            f"color: {self.current_app_status.color_theme.button_text_color};",
            inactive_button_style=f"background-color: {self.current_app_status.color_theme.primary_color};"
            f"color: {self.current_app_status.color_theme.button_text_color};",
        )
        self.episode_button_group.optionsSelected.connect(self.episode_button_clicked)
        self.top_layout.addWidget(self.episode_button_group, stretch=6)

        self.run_label = QLabel(
            f"Visualizing Run {self.current_sim_result_index+1}:1 Total Runs",
            self.top_bar,
        )
        self.run_label.setAlignment(Qt.AlignCenter)
        self.run_label.setStyleSheet(
            f"""
            background-color: {self.current_app_status.color_theme.primary_color};
            font-weight: bold;
            color: {self.current_app_status.color_theme.text_color};
            """
        )
        self.top_layout.addWidget(self.run_label, stretch=1)

    def episode_button_clicked(self, options):
        if not self.results:
            return  # No results to show
        option = options[0]  # button group is single select
        match option:
            case "<<<":
                self.current_sim_result_index = 0
            case "<<":
                self.current_sim_result_index = max(
                    0, self.current_sim_result_index - 10
                )
            case "<":
                self.current_sim_result_index = max(
                    0, self.current_sim_result_index - 1
                )
            case ">":
                self.current_sim_result_index = min(
                    len(self.results) - 1, self.current_sim_result_index + 1
                )
            case ">>":
                self.current_sim_result_index = min(
                    len(self.results) - 1, self.current_sim_result_index + 10
                )
            case ">>>":
                self.current_sim_result_index = len(self.results) - 1
        self.run_label.setText(
            f"Visualizing Run {self.current_sim_result_index+1}:{len(self.results)} Total Runs"
        )
        current_run_results = self.results[self.current_sim_result_index]

        self.sim_visualizer_widget.visualize_results(current_run_results.copy())
        self.update_run_info(
            current_run_results.iteration_info_batch.time,
            current_run_results.iteration_info_batch.execution_time,
        )
        self.selectedResultChanged.emit(current_run_results.copy())

    def init_ui(self):
        self.left_widget = QWidget(self)
        self.left_layout = QVBoxLayout(self.left_widget)
        self.layout.addWidget(self.left_widget, stretch=7)

        self.init_top_bar()

        self.sim_visualizer_widget = SimulationVisualizerWidget(
            self.current_app_status.color_theme, parent=self
        )
        self.left_layout.addWidget(self.sim_visualizer_widget)

        self.right_widget = QWidget(self)
        self.right_layout = QVBoxLayout(self.right_widget)
        self.layout.addWidget(self.right_widget, stretch=1)

        self.right_widget.setStyleSheet(
            """
            QWidget {
                color: white;
                background-color: """
            + self.current_app_status.color_theme.secondary_color
            + """;
            }
            QLabel {
                font-size: 12px;
                margin-bottom: 2px;
                padding: 5px;
                font-weight: bold;
                 background-color: """
            + self.current_app_status.color_theme.primary_color
            + """;
            }
            QLineEdit {
                background-color: """
            + self.current_app_status.color_theme.primary_color
            + """;
                     padding: 5px;
                     font-size: 12px;
                     margin-bottom: 5px;
                 }
               """
        )

        # Create a horizontal layout for the sim_run_count
        self.sim_run_count_layout = QHBoxLayout()
        self.sim_run_count_label = QLabel("Simulations to run:", self.right_widget)
        self.sim_run_count_edit = QLineEdit(self)
        self.sim_run_count_edit.setText(str(self.sim_run_count.value))
        self.sim_run_count_edit.editingFinished.connect(self.update_run_count)

        # Add the label and edit field to the horizontal layout
        self.sim_run_count_layout.addWidget(self.sim_run_count_label)
        self.sim_run_count_layout.addWidget(self.sim_run_count_edit)

        # Visualization interval with a similar approach
        self.visualization_interval_layout = QHBoxLayout()
        self.visualization_interval_label = QLabel(
            "Visualization Interval:", self.right_widget
        )
        self.visualization_interval_edit = QLineEdit(self)
        self.visualization_interval_edit.setText(str(self.visualization_interval.value))
        self.visualization_interval_edit.editingFinished.connect(
            self.update_visualization_interval
        )

        self.visualization_interval_layout.addWidget(self.visualization_interval_label)
        self.visualization_interval_layout.addWidget(self.visualization_interval_edit)

        # Add horizontal layouts to the main right layout
        self.right_layout.addLayout(self.sim_run_count_layout)
        self.right_layout.addLayout(self.visualization_interval_layout)

        self.controller_params = self.simulation_info.get_controller_params(
            self.simulation_info.controller_name,
            self.simulation_info.controller_params_name,
            self.current_app_status,
        )
        self.params_form = GenericFormWidget(
            self.controller_params,
            self.simulation_info.controller_name,
            color_theme=ColorTheme(
                primary_color=self.current_app_status.color_theme.primary_color,
                secondary_color=self.current_app_status.color_theme.primary_color,
                background_color=self.current_app_status.color_theme.primary_color,
                selected_color=self.current_app_status.color_theme.selected_color,
            ),
        )

        self.params_form.formChanged.connect(self.controller_params_changed)
        self.right_layout.addWidget(self.params_form, alignment=Qt.AlignTop)

        self.fps_label = QLabel("Avg FPS: N/A", self.right_widget)
        self.avg_time_label = QLabel("Avg Time Iteration: N/A", self.right_widget)
        self.max_time_label = QLabel("Max Time Iteration: N/A", self.right_widget)
        self.min_time_label = QLabel("Min Time Iteration: N/A", self.right_widget)
        self.total_exec_time_label = QLabel(
            "Total Execution Time: N/A", self.right_widget
        )
        self.total_time_label = QLabel("Total Sim Time: N/A", self.right_widget)

        self.right_layout.addWidget(self.fps_label)
        self.right_layout.addWidget(self.avg_time_label)
        self.right_layout.addWidget(self.max_time_label)
        self.right_layout.addWidget(self.min_time_label)
        self.right_layout.addWidget(self.total_exec_time_label)
        self.right_layout.addWidget(self.total_time_label)

        if self.errors:
            for error in self.errors:
                error_label = QLabel(error, self.right_widget)
                self.right_layout.addWidget(error_label)
        else:
            self.right_layout.addWidget(QLabel("No errors", self.right_widget))

        self.right_layout.addStretch(1)

    def controller_params_changed(self, params):
        self.params_pipe.send(params)

    def update_run_count(self):
        if self.sim_run_count_edit.text() == "":
            return
        try:
            new_value = int(self.sim_run_count_edit.text())
            if new_value > self.sim_process.completed_runs.value:
                self.sim_run_count.value = new_value
        except ValueError:
            # Optionally, reset the QLineEdit to a valid state or display an error message
            error_message = f" Invalid input: {self.sim_run_count_edit.text()}"
            self.sim_run_count_edit.setText(error_message)

    def update_visualization_interval(self):
        if self.visualization_interval_edit.text() == "":
            return
        try:
            new_value = int(self.visualization_interval_edit.text())
            self.visualization_interval.value = new_value
        except ValueError:
            error_message = f" Invalid input: {self.visualization_interval_edit.text()}"
            self.visualization_interval_edit.setText(error_message)

    def update_run_info(self, times: np.ndarray, execution_times: np.ndarray):
        if execution_times is None or execution_times.size == 0:
            return
        total_time = times[-1]

        total_exec_time = np.sum(execution_times)
        avg_time = total_exec_time / len(execution_times)
        max_time = np.max(execution_times)
        min_time = np.min(execution_times)
        fps = 1 / avg_time if avg_time > 0 else 0

        self.fps_label.setText(f"Avg FPS: {fps:.2f}")
        self.avg_time_label.setText(f"Avg Time Iteration: {avg_time:.4f} s")
        self.max_time_label.setText(f"Max Time Iteration: {max_time:.4f} s")
        self.min_time_label.setText(f"Min Time Iteration: {min_time:.4f} s")
        self.total_exec_time_label.setText(
            f"Total Execution Time: {self.format_time(total_exec_time)}"
        )
        self.total_time_label.setText(f"Total Sim Time: {self.format_time(total_time)}")

    def format_time(self, time_in_seconds):
        if time_in_seconds >= 60:
            return f"{time_in_seconds / 60:.2f} min"
        else:
            return f"{time_in_seconds:.4f} s"