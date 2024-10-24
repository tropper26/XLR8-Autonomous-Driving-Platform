import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from application.application_status import ApplicationStatus
from dto.waypoint import Waypoint
from local_planner.global_trajectory_planner import GlobalTrajectoryPlanner
from application.core.flow_layout import FlowLayout


class CanvasWithTrajectoryDto:
    def __init__(self, canvas, trajectory_df):
        self.canvas = canvas
        self.df = trajectory_df


class TrajectoryPlannerTab(QWidget):
    def __init__(self, parent, current_app_status: ApplicationStatus):
        super().__init__(parent=parent)

        self.current_app_status = current_app_status

        self.trajectory_planner = GlobalTrajectoryPlanner(
            self.current_app_status.planner_sampling_time
        )

        self.setStyleSheet(
            f"background-color: {current_app_status.color_theme.background_color};"
        )

        self.canvas_with_trajectory_dtos: dict[str, CanvasWithTrajectoryDto] = {}

        self.layout = QHBoxLayout(self)
        self.init_plots()

        self.click_plot_canvas(
            None, self.current_app_status.planning_strategy_name
        )  # Set the selected canvas to the saved strategy

    def init_plots(self):
        scroll_area = QScrollArea()
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setWidgetResizable(True)
        self.layout.addWidget(scroll_area)

        flow_widget = QWidget()
        flow_widget.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
        )
        flow_widget.setStyleSheet(
            f"background-color: {self.current_app_status.color_theme.secondary_color};"
        )
        flow_layout = FlowLayout(flow_widget)
        scroll_area.setWidget(flow_widget)

        self.widget_width = 400  # ~ 1/4 of the screen width
        self.widget_height = self.widget_width

        self.canvas_with_trajectory_dtos = {}

        canvas = self.create_canvas("Center of Path Spiral")
        self.canvas_with_trajectory_dtos[
            "Center of Path Spiral"
        ] = CanvasWithTrajectoryDto(canvas=canvas, trajectory_df=None)

        flow_layout.addWidget(canvas)

        # Pre-create Figure and FigureCanvas for each strategy
        for strategy_name in self.current_app_status.planning_strategy_options:
            canvas = self.create_canvas(strategy_name)

            self.canvas_with_trajectory_dtos[strategy_name] = CanvasWithTrajectoryDto(
                canvas=canvas, trajectory_df=None
            )

            flow_layout.addWidget(canvas)

    def create_canvas(self, strategy_name: str):
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.2, left=0.245)
        fig.set_size_inches(
            self.widget_width / fig.dpi,
            self.widget_height / fig.dpi,
        )
        ax.plot([], [], "b", label="Path")
        ax.plot([], [], "xr", label="Ref Points")
        ax.grid(True)
        ax.axis("equal")
        ax.set_xlabel("X[m]")
        ax.set_ylabel("Y[m]")
        ax.legend()

        figure_canvas = FigureCanvas(fig)
        figure_canvas.figure.axes[0].set_title(strategy_name)

        figure_canvas.enterEvent = (
            lambda event, name=strategy_name: self.enter_plot_canvas(event, name)
        )
        figure_canvas.leaveEvent = (
            lambda event, name=strategy_name: self.leave_plot_canvas(event, name)
        )
        figure_canvas.mousePressEvent = (
            lambda event, name=strategy_name: self.click_plot_canvas(event, name)
        )
        figure_canvas.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
        )
        figure_canvas.resize(self.widget_width, self.widget_height)

        return figure_canvas

    def click_plot_canvas(self, event, strategy_name: str):
        newly_selected_canvas_with_trajectory = self.canvas_with_trajectory_dtos[
            strategy_name
        ]

        if newly_selected_canvas_with_trajectory.df is None:
            print("No trajectory data for this strategy")
            return

        # Reset the color of the previously clicked canvas
        currently_selected_figure_canvas = self.canvas_with_trajectory_dtos[
            self.current_app_status.planning_strategy_name
        ].canvas
        fig = currently_selected_figure_canvas.figure
        fig.patch.set_facecolor(self.current_app_status.color_theme.background_color)
        fig.canvas.draw()

        # Change the color of the newly clicked canvas
        fig = newly_selected_canvas_with_trajectory.canvas.figure

        fig.patch.set_facecolor(self.current_app_status.color_theme.selected_color)
        fig.canvas.draw()

        self.current_app_status.planning_strategy_name = strategy_name
        self.current_app_status.ref_path_df = newly_selected_canvas_with_trajectory.df

    def enter_plot_canvas(self, event, strategy_name):
        if self.current_app_status.planning_strategy_name != strategy_name:
            canvas_with_trajectory_dto = self.canvas_with_trajectory_dtos[strategy_name]
            if canvas_with_trajectory_dto.df is not None:
                canvas_with_trajectory_dto.canvas.figure.patch.set_facecolor(
                    self.current_app_status.color_theme.hover_color
                )
                canvas_with_trajectory_dto.canvas.figure.canvas.draw()

    def leave_plot_canvas(self, event, strategy_name):
        if self.current_app_status.planning_strategy_name != strategy_name:
            canvas_with_trajectory_dto = self.canvas_with_trajectory_dtos[strategy_name]
            if canvas_with_trajectory_dto.df is not None:
                canvas_with_trajectory_dto.canvas.figure.patch.set_facecolor(
                    self.current_app_status.color_theme.background_color
                )
                canvas_with_trajectory_dto.canvas.figure.canvas.draw()

    def check_for_changes(self):
        waypoints = self.current_app_status.selected_route
        if not waypoints:
            return

        self.canvas_with_trajectory_dtos[
            "Center of Path Spiral"
        ].df = self.current_app_status.ref_path_df

        for strategy_name in self.current_app_status.planning_strategy_options:
            trajectory_df = self.trajectory_planner.generate_reference_trajectory(
                waypoints=waypoints, trajectory_strategy_name=strategy_name
            )

            self.canvas_with_trajectory_dtos[strategy_name].df = trajectory_df

        self.update_canvases(waypoints)

    def update_canvases(self, ref_points: list[Waypoint]):
        for (
            trajectory_strategy_name,
            canvas_with_trajectory,
        ) in self.canvas_with_trajectory_dtos.items():
            fig = canvas_with_trajectory.canvas.figure
            ax = fig.gca()

            trajectory_plot = ax.lines[0]

            ref_points_plot = ax.lines[1]

            if canvas_with_trajectory.df is None:
                trajectory_plot.set_data([], [])
                ref_points_plot.set_data([], [])
            else:
                trajectory_plot.set_data(
                    canvas_with_trajectory.df.X, canvas_with_trajectory.df.Y
                )
                ref_points_plot.set_data(
                    [waypoint.x for waypoint in ref_points],
                    [waypoint.y for waypoint in ref_points],
                )

            ax.relim()
            ax.autoscale_view()
            trajectory_plot.figure.canvas.draw()