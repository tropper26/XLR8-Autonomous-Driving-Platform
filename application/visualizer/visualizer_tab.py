import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QScrollArea, QHBoxLayout
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QWidget,
)

from application.application_status import ApplicationStatus
from application.visualizer.widgets.error_plotter_widget import ErrorPlotterWidget
from application.visualizer.widgets.simulation_visualizer_widget import (
    SimulationVisualizerWidget,
)
from dto.coord_transform import compute_path_frame_error


class VisualizerTab(QWidget):
    def __init__(self, parent, current_app_status: ApplicationStatus):
        super().__init__(parent=parent)

        self.current_app_status = current_app_status
        self.sim_running = False
        self.layout = QVBoxLayout(self)

        self.init_ui()

    def init_ui(self):
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
        )  # Options: Qt.ScrollBarAlwaysOff, Qt.ScrollBarAsNeeded, Qt.ScrollBarAlwaysOn

        self.vertical_widget = QWidget(self.scroll_area)
        self.vertical_widget.setStyleSheet(
            f"background-color: {self.current_app_status.color_theme.background_color};"
        )
        self.vertical_layout = QVBoxLayout(self.vertical_widget)

        self.scroll_area.setWidget(self.vertical_widget)

        self.layout.addWidget(self.scroll_area)

    def reset(self):
        # remove all widgets from the layout
        for i in reversed(range(self.vertical_layout.count())):
            widget = self.vertical_layout.itemAt(i).widget()
            widget.deleteLater()

    def preprocess_rewards(self, rewards: np.ndarray):
        """Preprocess rewards to be visualized."""

        return np.clip(
            rewards, 0, 5
        )  # 0-5 range of the reward function, outliers excluded

    def init_visualization(self):
        simulation_results = list(self.current_app_status.simulation_results.values())

        self.sim_visualizer_widget = SimulationVisualizerWidget(
            self.current_app_status.color_theme, parent=self
        )
        self.sim_visualizer_widget.setFixedHeight(1200)
        self.vertical_layout.addWidget(self.sim_visualizer_widget)

        self.sim_visualizer_widget.setup_ref_path(
            self.current_app_status.ref_path_df,
            self.current_app_status.path_obstacles,
            self.current_app_status.world_x_limit,
            self.current_app_status.world_y_limit,
        )
        self.sim_visualizer_widget.visualize_results(simulation_results.copy())

        Crosstrack_errors = [
            (sim_result.iteration_infos.time, sim_result.iteration_infos.error_Y)
            for sim_result in simulation_results
        ]

        Heading_errors = [
            (sim_result.iteration_infos.time, sim_result.iteration_infos.error_Psi)
            for sim_result in simulation_results
        ]

        Velocity_errors = [
            (sim_result.iteration_infos.time, sim_result.iteration_infos.error_x_dot)
            for sim_result in simulation_results
        ]

        Scores = [
            (
                sim_result.iteration_infos.time,
                self.preprocess_rewards(sim_result.iteration_infos.reward),
            )
            for sim_result in simulation_results
        ]

        Crosstrack_errors_path = []
        Heading_errors_path = []
        Path_Scores = []
        for sim_result in simulation_results:
            (
                _,
                error_Y_path_frame,
                error_psi,
                _,
            ) = compute_path_frame_error(
                X_path=sim_result.iteration_infos.X_path.values,
                Y_path=sim_result.iteration_infos.Y_path.values,
                Psi_path=sim_result.iteration_infos.Psi_path.values,
                X_vehicle=sim_result.iteration_infos.X.values,
                Y_vehicle=sim_result.iteration_infos.Y.values,
                Psi_vehicle=sim_result.iteration_infos.Psi.values,
            )

            delta_d = np.diff(sim_result.iteration_infos.d)
            delta_d = np.insert(delta_d, 0, 0.0)
            path_rewards = (
                self.current_app_status.reward_manager.compute_reward_without_velocity(
                    error_Y_path_frame, error_psi, delta_d
                )
            )

            Crosstrack_errors_path.append(
                (sim_result.iteration_infos.time, error_Y_path_frame)
            )
            Heading_errors_path.append((sim_result.iteration_infos.time, error_psi))

            Path_Scores.append((sim_result.iteration_infos.time, path_rewards))

        names = [
            sim_result.simulation_info.identifier for sim_result in simulation_results
        ]

        horiz_widget = QWidget(self.vertical_widget)
        horiz_layout = QHBoxLayout(horiz_widget)
        self.vertical_layout.addWidget(horiz_widget)

        self.Crosstrack_error_plotter_widget = ErrorPlotterWidget(
            color_theme=self.current_app_status.color_theme,
            title="Vehicle - Trajectory Crosstrack Error Plot",
            trajectories=Crosstrack_errors,
            names=names,
            y_axis_label="CTE [m]",
        )
        self.Crosstrack_error_plotter_widget.setFixedHeight(450)
        horiz_layout.addWidget(self.Crosstrack_error_plotter_widget, stretch=1)

        self.Crosstrack_error_path_plotter_widget = ErrorPlotterWidget(
            color_theme=self.current_app_status.color_theme,
            title="Vehicle - Path Crosstrack Error Plot",
            trajectories=Crosstrack_errors_path,
            names=names,
            y_axis_label="CTE [m]",
            y_upper_bound=self.current_app_status.reward_manager.max_lateral_error_threshold,
            y_lower_bound=-self.current_app_status.reward_manager.max_lateral_error_threshold,
        )
        self.Crosstrack_error_path_plotter_widget.setFixedHeight(450)

        horiz_layout.addWidget(self.Crosstrack_error_path_plotter_widget, stretch=1)

        horiz_widget_2 = QWidget(self.vertical_widget)
        horiz_layout_2 = QHBoxLayout(horiz_widget_2)
        self.vertical_layout.addWidget(horiz_widget_2)

        self.Heading_error_plotter_widget = ErrorPlotterWidget(
            color_theme=self.current_app_status.color_theme,
            title="Vehicle - Trajectory Heading Error Plot",
            trajectories=Heading_errors,
            names=names,
            y_axis_label="Heading [rad]",
        )
        self.Heading_error_plotter_widget.setFixedHeight(450)
        horiz_layout_2.addWidget(self.Heading_error_plotter_widget, stretch=1)
        self.vertical_layout.addWidget(horiz_widget_2)

        self.Heading_error_path_plotter_widget = ErrorPlotterWidget(
            color_theme=self.current_app_status.color_theme,
            title="Vehicle - Path Heading Error Plot",
            trajectories=Heading_errors_path,
            names=names,
            y_axis_label="Heading [rad]",
            y_lower_bound=-self.current_app_status.reward_manager.max_heading_error_threshold,
            y_upper_bound=self.current_app_status.reward_manager.max_heading_error_threshold,
        )
        self.Heading_error_path_plotter_widget.setFixedHeight(450)
        horiz_layout_2.addWidget(self.Heading_error_path_plotter_widget, stretch=1)

        horiz_widget_3 = QWidget(self.vertical_widget)
        horiz_layout_3 = QHBoxLayout(horiz_widget_3)
        self.vertical_layout.addWidget(horiz_widget_3)

        self.Rewards_plotter_widget = ErrorPlotterWidget(
            color_theme=self.current_app_status.color_theme,
            title="Vehicle - Trajectory Performance Plot",
            trajectories=Scores,
            names=names,
            y_axis_label="Score",
            y_upper_bound=5,
            y_lower_bound=0,
        )
        self.Rewards_plotter_widget.setFixedHeight(450)
        horiz_layout_3.addWidget(self.Rewards_plotter_widget, stretch=1)

        self.Velocity_error_plotter_widget = ErrorPlotterWidget(
            color_theme=self.current_app_status.color_theme,
            title="Vehicle - Velocity Profile Error Plot",
            trajectories=Velocity_errors,
            names=names,
            y_axis_label="Velocity [m/s]",
        )
        self.Velocity_error_plotter_widget.setFixedHeight(450)
        horiz_layout_3.addWidget(self.Velocity_error_plotter_widget)

        horiz_widget_4 = QWidget(self.vertical_widget)
        horiz_layout_4 = QHBoxLayout(horiz_widget_4)
        self.vertical_layout.addWidget(horiz_widget_4)

        self.Path_Rewards_plotter_widget = ErrorPlotterWidget(
            color_theme=self.current_app_status.color_theme,
            title="Vehicle - Path Performance Plot",
            trajectories=Path_Scores,
            names=names,
            y_axis_label="Score",
            y_upper_bound=5,
            y_lower_bound=0,
        )

        self.Path_Rewards_plotter_widget.setFixedHeight(450)
        horiz_layout_4.addWidget(self.Path_Rewards_plotter_widget, stretch=1)

        progress_along_track = [
            (sim_result.iteration_infos.time, sim_result.iteration_infos.S_path)
            for sim_result in simulation_results
        ]

        self.Progress_along_track_plotter_widget = ErrorPlotterWidget(
            color_theme=self.current_app_status.color_theme,
            title="Vehicle - Progress Along Path Plot",
            trajectories=progress_along_track,
            names=names,
            y_axis_label="Distance [m]",
        )
        self.Progress_along_track_plotter_widget.setFixedHeight(450)
        horiz_layout_4.addWidget(self.Progress_along_track_plotter_widget)