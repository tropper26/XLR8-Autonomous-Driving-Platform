import math

import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt
from matplotlib import gridspec, patches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.animation import FuncAnimation
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QPushButton,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
)
from control.controller_viz_info import ControllerVizInfo, Types
from matplotlib.patches import Rectangle as pltRectangle, Polygon

from dto.color_theme import ColorTheme
from dto.geometry import Rectangle
from parametric_curves.path_segment import PathSegmentDiscretization
from simulation.iteration_info import IterationInfo, IterationInfoBatch
from simulation.simulation_result import SimulationResult
from state_space.states.state import State
from vehicle.vehicle_params import VehicleParams


def update_axes_limits(
        axes, padding, x_lim: (float, float) = None, y_lim: (float, float) = None
):
    """
    Recalculates the data limits and autoscales the given axes, then adds specified padding to the axes' limits.

    Args:
    axes (matplotlib.axes.Axes): The matplotlib axes to update.
    padding (float): The amount of padding to add to the axes' x and y limits.
    """
    if x_lim and y_lim:
        axes.set_xlim(x_lim[0] - padding, x_lim[1] + padding)
        axes.set_ylim(y_lim[0] - padding, y_lim[1] + padding)
    else:
        # Recalculate the data limits based on the current data linked to the axes
        axes.relim()
        axes.autoscale_view()

        # Get current axes limits
        x_min, x_max = axes.get_xlim()
        y_min, y_max = axes.get_ylim()

        # Set new axes limits with additional padding
        axes.set_xlim(x_min - padding, x_max + padding)
        axes.set_ylim(y_min - padding, y_max + padding)


class SimulationVisualizerWidget(QWidget):
    def __init__(
            self,
            color_theme: ColorTheme,
            parent=None,
    ):
        super().__init__(parent)
        self.invalid_trajectories_plot = None
        self.alternate_trajectories_plot = None
        self.reference_trajectory_plot = None
        self.path_obstacles = None
        self.ref_path = None
        self.gs = None
        self.table = None
        self.color_theme = color_theme
        self.controller_viz_ref_plots = None
        self.reward_exp_label = None
        self.iteration_info_layout = None
        self.iteration_info_widget = None
        self.columns_to_display = None
        self.column_count = None
        self.ref_Y_plot = None
        self.ref_X_plot = None
        self.ref_psi_plot = None
        self.ref_x_dot_plot = None
        self.max_frame_count = None
        self.new_sim_results = None
        self.controller_viz_plots = None
        self.main_plot_ax = None
        self.animation = None
        self.current_index = None
        self.current_sim_results_to_visualize: list[SimulationResult] = []
        self.car_trajectory_plots = None
        self.canvas = None
        self.main_plot_fig = None
        self.play_button = None
        self.Y_plot = None
        self.X_plot = None
        self.psi_plot = None
        self.x_dot_plot = None

        # Initialize the car shape and path plots
        self.vehicle_polygons = []
        self.visual_range_ellipses = []
        self.wheel_polygons = []
        self.car_trajectory_plots = []
        self.controller_viz_plots = []
        self.controller_viz_ref_plots = []

        plt.style.use(
            {
                "axes.edgecolor": "black",
                "axes.labelcolor": "black",
                "text.color": "black",
                "xtick.color": "black",
                "ytick.color": "black",
            }
        )

        self.vehicle_colors = [
            "cyan",
            "yellow",
            "magenta",
        ]
        self.trajectory_colors = [
            "blue",
            "orange",
            "purple",
        ]

        self.vehicle_front_wheel_color = "gray"
        self.central_line_color = "gray"
        self.path_color = "black"
        self.obstacle_color = "gray"
        self.alternate_trajectory_color = "green"
        self.invalid_trajectory_color = "red"

        self.buttons_enabled = False
        self.playing = False
        self.vehicle_plot_count = 0
        self.sim_results_counter = (
            0  # Incremented every time a new sim result is received
        )
        self.last_sim_nr = 1  # The last sim number that was visualized

        self.setup_ui()

    def setup_ui(self):
        self.setStyleSheet(f"background-color: {self.color_theme.primary_color};")
        main_vertical_layout = QVBoxLayout(self)
        self.setup_main_container(main_vertical_layout, stretch=9)
        self.setup_button_bar(main_vertical_layout, stretch=1)

        self.iteration_info_widget = QWidget(parent=self)
        self.iteration_info_layout = QVBoxLayout(self.iteration_info_widget)
        main_vertical_layout.addWidget(self.iteration_info_widget)

        self.reward_exp_label = QLabel(
            "Reward Explanation: ", self.iteration_info_widget
        )
        self.reward_exp_label.setStyleSheet(f"color: {self.color_theme.text_color}")
        self.iteration_info_layout.addWidget(self.reward_exp_label)

        self.table = QTableWidget(self.iteration_info_widget)
        self.table.setRowCount(5)
        self.columns_to_display = [
            "a",
            "d",
            "X",
            "X_ref",
            "error_X",
            "Y",
            "Y_ref",
            "error_Y",
            "Psi",
            "Psi_ref",
            "error_Psi",
            "x_dot",
            "x_dot_ref",
            "error_x_dot",
            "y_dot",
            "psi_dot",
            "reward",
            "time",
            "S_ref",
            "K_ref",
        ]

        self.column_count = len(self.columns_to_display)
        self.table.setColumnCount(self.column_count)
        self.table.setHorizontalHeaderLabels(self.columns_to_display)

        self.iteration_info_layout.addWidget(self.table)
        self.iteration_info_layout.addStretch(1)

    def setup_button_bar(self, layout, stretch):
        button_bar_widget = QWidget(self)
        button_bar_layout = QHBoxLayout(button_bar_widget)
        layout.addWidget(button_bar_widget, stretch=stretch)

        first_button = QPushButton("First", button_bar_widget)
        first_button.setStyleSheet("background-color: red;")
        first_button.clicked.connect(self.navigate_first)

        prev_button = QPushButton("Previous", button_bar_widget)
        prev_button.setStyleSheet("background-color: red;")
        prev_button.clicked.connect(self.navigate_previous)

        next_button = QPushButton("Next", button_bar_widget)
        next_button.setStyleSheet("background-color: red;")
        next_button.clicked.connect(self.navigate_next)

        self.play_button = QPushButton("Pause", button_bar_widget)
        self.play_button.setStyleSheet("background-color: red;")
        self.play_button.clicked.connect(self.toggle_play)

        last_button = QPushButton("Last", button_bar_widget)
        last_button.setStyleSheet("background-color: red;")
        last_button.clicked.connect(self.navigate_last)

        reset_button = QPushButton("Reset", button_bar_widget)
        reset_button.setStyleSheet("background-color: red;")
        reset_button.clicked.connect(self.reset)

        button_bar_layout.addWidget(first_button)
        button_bar_layout.addWidget(prev_button)
        button_bar_layout.addWidget(self.play_button)
        button_bar_layout.addWidget(next_button)
        button_bar_layout.addWidget(last_button)
        button_bar_layout.addWidget(reset_button)

    def setup_main_container(self, layout, stretch):
        main_container_widget = QWidget(self)
        main_container_layout = QVBoxLayout(main_container_widget)
        layout.addWidget(main_container_widget, stretch=stretch)


        # self.main_plot_fig = plt.figure(
        #     figsize=(16, 9), dpi=120, facecolor=(0.8, 0.8, 0.8)
        # )
        # self.gs = gridspec.GridSpec(12, 12)
        #
        # self.setup_sub_plots()
        #
        # self.main_plot_ax = self.main_plot_fig.add_subplot(self.gs[3:, :])
        # self.main_plot_ax.set_facecolor((0.9, 0.9, 0.9))


        self.main_plot_fig = plt.figure(
            figsize=(16, 9), dpi=120, facecolor=(0.8, 0.8, 0.8)
        )
        self.main_plot_ax = self.main_plot_fig.add_subplot(1, 1, 1)
        self.main_plot_ax.set_facecolor((0.9, 0.9, 0.9))



        main_plot_widget = QWidget(self)
        main_plot_layout = QVBoxLayout(main_plot_widget)
        main_container_layout.addWidget(main_plot_widget)

        self.canvas = FigureCanvas(self.main_plot_fig)
        self.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.canvas.setFocusPolicy(Qt.ClickFocus)
        self.canvas.setFocus()

        # Disable specific key bindings
        plt.rcParams["keymap.fullscreen"] = []  # Disables the 'f' key for fullscreen
        plt.rcParams["keymap.home"] = ["h"]  # Disables the 'r', 'h' and 'home' keys
        plt.rcParams["keymap.back"] = ["backspace"]  # Disables the 'left' key
        plt.rcParams["keymap.forward"] = []  # Disables the 'right' key

        toolbar = NavigationToolbar(self.canvas, main_plot_widget)
        main_plot_layout.addWidget(toolbar, stretch=1)
        main_plot_layout.addWidget(self.canvas, stretch=9)

    def on_key_press(self, event):
        if self.buttons_enabled:
            if event.key == "down":
                self.navigate_first()
            elif event.key == "left":
                self.navigate_previous()
            elif event.key == " ":
                self.toggle_play()
            elif event.key == "right":
                self.navigate_next()
            elif event.key == "up":
                self.navigate_last()
            elif event.key == "r":
                self.reset()

    def setup_sub_plots(self):
        # Adjust indices for top row placement
        row_start = 0
        row_end = 2
        self.ref_X_plot, self.X_plot = self.create_subplot_for_main_plot(
            self.main_plot_fig, self.gs, row_start, row_end, 0, 3, "X [m]"
        )
        self.ref_Y_plot, self.Y_plot = self.create_subplot_for_main_plot(
            self.main_plot_fig, self.gs, row_start, row_end, 3, 6, "Y [m]"
        )
        self.ref_psi_plot, self.psi_plot = self.create_subplot_for_main_plot(
            self.main_plot_fig, self.gs, row_start, row_end, 6, 9, "Psi [rad]"
        )
        self.ref_x_dot_plot, self.x_dot_plot = self.create_subplot_for_main_plot(
            self.main_plot_fig, self.gs, row_start, row_end, 9, 12, "x_dot [m/s]"
        )
        self.main_plot_fig.subplots_adjust(
            left=0.035, bottom=0.05, right=0.965, top=0.95, hspace=0.20, wspace=1.1
        )

    def create_subplot_for_main_plot(
            self,
            fig,
            gs,
            row_start,
            row_end,
            col_start,
            col_end,
            ylabel,
    ):
        ax = fig.add_subplot(
            gs[row_start:row_end, col_start:col_end], facecolor=(0.9, 0.9, 0.9)
        )
        (ref_plot_to_be_animated,) = ax.plot(
            [], [], "-", linewidth=1, color=self.path_color
        )
        (plot_to_be_animated,) = ax.plot(
            [],
            [],
            "-",
            linewidth=1,
            color=self.trajectory_colors[self.vehicle_plot_count],
        )
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.xlabel(ylabel, fontsize=10)  # To fit the relevant label on the screen
        ax.xaxis.set_label_position("top")
        ax.yaxis.tick_right()
        ax.grid(True)

        return ref_plot_to_be_animated, plot_to_be_animated

    def setup_legend(self):
        legend = self.main_plot_ax.legend(
            fancybox=True,
            framealpha=0.5,
            shadow=True,
            borderpad=1,
            fontsize="small",
            handlelength=2,
            facecolor="white",
        )
        for text in legend.get_texts():
            text.set_color("black")

    def init_subplots(self, iteration_info_batch: IterationInfoBatch):
        self.ref_X_plot.set_data(iteration_info_batch.S_ref, iteration_info_batch.X_ref)
        update_axes_limits(self.ref_X_plot.axes, padding=0.5)

        self.ref_Y_plot.set_data(iteration_info_batch.S_ref, iteration_info_batch.Y_ref)
        update_axes_limits(self.ref_Y_plot.axes, padding=0.5)

        self.ref_psi_plot.set_data(
            iteration_info_batch.S_ref, iteration_info_batch.Psi_ref
        )
        update_axes_limits(self.ref_psi_plot.axes, padding=0.5)

        self.ref_x_dot_plot.set_data(
            iteration_info_batch.S_ref, iteration_info_batch.x_dot_ref
        )
        update_axes_limits(self.ref_x_dot_plot.axes, padding=0.5)

    def setup_ref_path(
            self,
            discretized_path: PathSegmentDiscretization,
            path_obstacles: list[Rectangle],
            simulation_x_limit: float,
            simulation_y_limit: float,
    ):
        self.main_plot_ax.clear()

        self.ref_path = discretized_path
        self.path_obstacles = path_obstacles

        for rect in path_obstacles:
            patch = pltRectangle(
                (rect.x, rect.y),
                rect.width,
                rect.height,
                linewidth=1,
                edgecolor=self.obstacle_color,
                facecolor=self.obstacle_color,
            )
            self.main_plot_ax.add_patch(patch)

        self.main_plot_ax.plot(
            discretized_path.X,
            discretized_path.Y,
            "--",
            linewidth=1,
            label="Central Line",
            color=self.central_line_color,
        )

        for i in range(0, discretized_path.lateral_X.shape[1], 2):
            lateral_X = discretized_path.lateral_X[:, i]
            lateral_Y = discretized_path.lateral_Y[:, i]

            self.main_plot_ax.plot(
                lateral_X,
                lateral_Y,
                "-",
                linewidth=3,
                color=self.path_color,
            )

        (self.reference_trajectory_plot,) = self.main_plot_ax.plot(
            [],
            [],
            "-",
            color=self.trajectory_colors[self.vehicle_plot_count],
            zorder=10,
            label="Optimal Trajectory",
        )
        self.alternate_trajectories_plot = self.main_plot_ax.scatter(
            [],
            [],
            c=self.alternate_trajectory_color,
            s=1,
            zorder=9,
            label="Alternate Trajectories",
        )

        self.invalid_trajectories_plot = self.main_plot_ax.scatter(
            [],
            [],
            c=self.invalid_trajectory_color,
            s=1,
            zorder=8,
            label="Invalid Trajectories",
        )

        self.setup_legend()
        self.main_plot_ax.grid(True)
        self.main_plot_ax.title.set_text("Run: 1")
        update_axes_limits(
            self.main_plot_ax,
            padding=1,
            x_lim=(0, simulation_x_limit),
            y_lim=(0, simulation_y_limit),
        )

        self.canvas.draw_idle()

    def add_visual_area_plot(self):
        visual_area_ellipse = patches.Ellipse(
            (0, 0),
            width=10,
            height=5,
            angle=0,
            fill=False,
            edgecolor="purple",
            zorder=20,
            label="Visual Range of Vehicle",
        )

        self.main_plot_ax.add_patch(visual_area_ellipse)
        self.visual_range_ellipses.append(visual_area_ellipse)

    def add_car_plot(self, vp: VehicleParams):
        name = f"{self.vehicle_plot_count}" if self.vehicle_plot_count > 0 else ""

        vehicle_polygon = plt.Polygon(
            [(0, 0), (0.1, 0.1)],  # temporary values
            closed=True,
            fill=True,
            zorder=20,
            color=self.vehicle_colors[self.vehicle_plot_count],
            label=f"Vehicle {name}",
        )
        self.main_plot_ax.add_patch(vehicle_polygon)
        self.vehicle_polygons.append(vehicle_polygon)
        if vp.is_bicycle():
            wheel_count = 2
        else:
            wheel_count = 4

        wheel_polygons = []
        for _ in range(wheel_count):
            wheel_polygon = plt.Polygon(
                [(0, 0), (0.1, 0.1)],  # temporary values
                closed=True,
                fill=True,
                zorder=21,
                color=self.vehicle_front_wheel_color,
            )
            self.main_plot_ax.add_patch(wheel_polygon)
            wheel_polygons.append(wheel_polygon)
        self.wheel_polygons.append(wheel_polygons)

        (car_trajectory_plot,) = self.main_plot_ax.plot(
            [],
            [],
            linestyle="-",
            color=self.trajectory_colors[self.vehicle_plot_count],
            linewidth=1,
            zorder=5,
            label=f"Previous Trajectory {name}",
        )
        self.car_trajectory_plots.append(car_trajectory_plot)

        (controller_viz_plot,) = self.main_plot_ax.plot(
            [],
            [],
            color=self.vehicle_colors[-1],  # temporary color
            alpha=0.7,
            linewidth=1,
            zorder=7,
        )
        self.controller_viz_plots.append(controller_viz_plot)

        (controller_viz_ref_plot,) = self.main_plot_ax.plot(
            [],
            [],
            color=self.trajectory_colors[-1],  # temporary color
            alpha=0.3,
            zorder=6,
            linewidth=1,
        )
        self.vehicle_plot_count += 1

        self.controller_viz_ref_plots.append(controller_viz_ref_plot)

        self.setup_legend()

    def visualize_results(self, new_sim_res: list[SimulationResult]):
        if isinstance(new_sim_res, SimulationResult):
            new_sim_res = [new_sim_res]

        self.sim_results_counter += 1

        for index in range(len(new_sim_res) - self.vehicle_plot_count):
            self.add_car_plot(new_sim_res[index].vehicle_params_for_visualization)

        self.add_visual_area_plot()

        self.new_sim_results = new_sim_res.copy()

        if not self.animation:  # First time visualization
            self.update_results_to_visualize(self.new_sim_results.copy())
            self.init_subplots(self.current_sim_results_to_visualize[0].iteration_info_batch)

            self.buttons_enabled = True
            self.playing = True

            self.animation = FuncAnimation(
                self.main_plot_fig,
                self.update_plots,
                frames=self.max_frame_count,  # if it's a number it does range(nr) if its a list it does the list
                interval=1,
                repeat=True,
                blit=False,  # blit=True is faster but doesn't work with rewind
            )

        self.canvas.draw()

    def update_results_to_visualize(self, new_sim_res: list[SimulationResult]) -> list[SimulationResult]:
        sim_results = []
        self.last_sim_nr = self.sim_results_counter

        frame_count = max(
            [len(sim.iteration_infos) for sim in new_sim_res]
        )
        animation_step = int(frame_count / 144) + 1
        for sim_result in self.new_sim_results:
            sim_result_to_animate = sim_result.down_sample_for_visualization(
                animation_step
            )
            sim_results.append(sim_result_to_animate)

        self.max_frame_count = max(
            [len(sim.iteration_infos) for sim in sim_results]
        )

        self.current_sim_results_to_visualize = sim_results

        return self.current_sim_results_to_visualize

    def on_animation_end(self):
        if (
                self.sim_results_counter > self.last_sim_nr
        ):  # New simulation results received
            self.update_results_to_visualize(self.new_sim_results.copy())

            self.main_plot_ax.title.set_text(
                f"Run: {self.current_sim_results_to_visualize[0].run_index + 1}"
            )

            self.init_subplots(self.current_sim_results_to_visualize[0].iteration_info_batch)

        if self.playing:
            self.current_index = 0

    def update_plots(self, index=None):
        if index is not None:
            self.current_index = index
        index = self.current_index

        vehicle_polygon: Polygon
        visual_range_ellipse: patches.Ellipse
        wheel_polygon: Polygon
        sim_result: SimulationResult

        for (
                vehicle_polygon,
                visual_range_ellipse,
                wheel_polygons,
                car_trajectory_plot,
                controller_viz_plot,
                controller_viz_ref_plot,
                sim_result,
        ) in zip(
            self.vehicle_polygons,
            self.visual_range_ellipses,
            self.wheel_polygons,
            self.car_trajectory_plots,
            self.controller_viz_plots,
            self.controller_viz_ref_plots,
            self.current_sim_results_to_visualize,
        ):
            if index < len(sim_result.iteration_infos):
                current_iteration = sim_result.iteration_infos[index]

                current_state = State(
                    current_iteration.X,
                    current_iteration.Y,
                    current_iteration.Psi,
                    current_iteration.x_dot,
                    current_iteration.y_dot,
                    current_iteration.psi_dot,
                )
                vehicle_shape, wheel_shapes = draw_vehicle(
                    current_state,
                    current_iteration.d,
                    sim_result.vehicle_params_for_visualization,
                )
                vehicle_polygon.set_xy(vehicle_shape)

                front_axle_X, front_axle_Y = sim_result.vehicle_params_for_visualization.front_axle_position(
                    current_state)
                visual_range_ellipse.set_center((front_axle_X, front_axle_Y))
                visual_range_ellipse.angle = np.degrees(current_state.Psi)

                for wheel_polygon, wheel_shape in zip(wheel_polygons, wheel_shapes):
                    wheel_polygon.set_xy(wheel_shape)

                car_trajectory_plot.set_data(
                    sim_result.iteration_info_batch.X[: index + 1],
                    sim_result.iteration_info_batch.Y[: index + 1],
                )

                if isinstance(current_iteration.controller_viz_info, ControllerVizInfo):
                    controller_viz_plot.set_data(
                        current_iteration.controller_viz_info.X,
                        current_iteration.controller_viz_info.Y,
                    )

                    controller_viz_ref_plot.set_data(
                        current_iteration.controller_viz_info.ref_X,
                        current_iteration.controller_viz_info.ref_Y,
                    )

                    if current_iteration.controller_viz_info.viz_type == Types.Line:
                        controller_viz_plot.set_linestyle("-")
                        controller_viz_ref_plot.set_linestyle("-")
                        controller_viz_plot.set_marker("o")
                        controller_viz_ref_plot.set_marker("o")
                    elif current_iteration.controller_viz_info.viz_type == Types.Point:
                        controller_viz_plot.set_linestyle("None")
                        controller_viz_ref_plot.set_linestyle("None")
                        controller_viz_plot.set_marker("x")
                        controller_viz_ref_plot.set_marker("x")

        sim_result = self.current_sim_results_to_visualize[
            0
        ]  # Currently display only for first simulation
        if index < len(sim_result.iteration_infos):
            current_iteration = sim_result.iteration_infos[index]

            S_ref_till_index = sim_result.iteration_info_batch.S_ref[: index + 1]
            self.X_plot.set_data(
                S_ref_till_index, sim_result.iteration_info_batch.X[: index + 1]
            )
            self.Y_plot.set_data(
                S_ref_till_index, sim_result.iteration_info_batch.Y[: index + 1]
            )
            self.x_dot_plot.set_data(
                S_ref_till_index, sim_result.iteration_info_batch.x_dot[: index + 1]
            )
            self.psi_plot.set_data(
                S_ref_till_index, sim_result.iteration_info_batch.psi_dot[: index + 1]
            )

            X = current_iteration.reference_trajectory.discretized.X
            Y = current_iteration.reference_trajectory.discretized.Y
            self.reference_trajectory_plot.set_data(X, Y)

            alternate_trajectories = current_iteration.alternate_trajectories
            invalid_trajectories = current_iteration.invalid_trajectories

            # YES WE DO NEED THESE EXTRA CHECKS
            if len(alternate_trajectories) > 0 or len(invalid_trajectories) > 0:
                if len(alternate_trajectories) > 0:
                    X = np.concatenate(
                        [at.discretized.X for at in alternate_trajectories]
                    )
                    Y = np.concatenate(
                        [at.discretized.Y for at in alternate_trajectories]
                    )

                    self.alternate_trajectories_plot.set_offsets(np.c_[X, Y])
                else:
                    self.alternate_trajectories_plot.set_offsets(np.c_[[], []])

                if len(invalid_trajectories) > 0:
                    X = np.concatenate(
                        [it.discretized.X for it in invalid_trajectories]
                    )
                    Y = np.concatenate(
                        [it.discretized.Y for it in invalid_trajectories]
                    )

                    self.invalid_trajectories_plot.set_offsets(np.c_[X, Y])
                else:
                    self.invalid_trajectories_plot.set_offsets(np.c_[[], []])

            self.reward_exp_label.setText(
                f"Reward Explanation: {current_iteration.reward_explanation}"
            )

            self.update_state_table(sim_result.iteration_infos, index)

            self.canvas.draw_idle()  # Redraw the canvas when manually updating the plots

        if index == self.max_frame_count - 1:
            self.on_animation_end()

        return iter(
            (
                self.vehicle_polygons,
                self.visual_range_ellipses,
                self.wheel_polygons,
                self.car_trajectory_plots,
                self.controller_viz_plots,
                self.controller_viz_ref_plots,
            )
        )

    def update_state_table(self, iteration_infos: list[IterationInfo], index: int):
        table = [
            iteration_info.as_table(self.columns_to_display)
            for iteration_info in iteration_infos
        ]

        if index == 0:
            self.clear_cells([0, 1], self.column_count)
            table_row_index_offset = 2
        elif index == 1:
            self.clear_cells([0], self.column_count)
            table_row_index_offset = 1
        elif index == self.max_frame_count - 2:
            self.clear_cells([4], self.column_count)
            table_row_index_offset = 0
        elif index == self.max_frame_count - 1:
            self.clear_cells([3, 4], self.column_count)
            table_row_index_offset = 0
        else:
            table_row_index_offset = 0

        start_index = max(index - 2, 0)
        end_index = min(index + 3, len(iteration_infos))
        # Update table content
        self.set_table_text(table, start_index, end_index, table_row_index_offset)

    def clear_cells(self, row_indices, column_count):
        """Clears text for specified rows."""
        for row in row_indices:
            for col in range(column_count):
                if self.table.item(row, col):
                    self.table.item(row, col).setText("")

    def set_table_text(
            self, table: list[list[float]], start_index: int, end_index: int, offset: int
    ):
        """Set table cells from DataFrame values with rounding for floats."""
        for i, row_idx in enumerate(range(start_index, end_index)):
            for j in range(self.column_count):
                value = table[row_idx][j]
                text = str(round(value, 2)) if isinstance(value, float) else str(value)
                if self.table.item(i + offset, j):
                    self.table.item(i + offset, j).setText(text)
                else:
                    self.table.setItem(i + offset, j, QTableWidgetItem(text))

    def navigate_first(self):
        if self.buttons_enabled and not self.playing:
            self.current_index = 0
            self.update_plots()

    def navigate_previous(self):
        if self.buttons_enabled and not self.playing and self.current_index > 0:
            self.current_index -= 1
            self.update_plots()

    def navigate_next(self):
        if (
                self.buttons_enabled
                and not self.playing
                and self.current_index < self.max_frame_count - 1
        ):
            self.current_index += 1
            self.update_plots()

    def navigate_last(self):
        if self.buttons_enabled and not self.playing:
            self.current_index = self.max_frame_count - 1
            self.update_plots()

    def reset(self):
        if self.buttons_enabled:
            self.on_animation_end()
            self.animation.frame_seq = self.animation.new_frame_seq()
            if not self.playing:
                self.toggle_play()

    def toggle_play(self):
        if self.buttons_enabled:
            if self.playing:
                self.animation.event_source.stop()
                self.play_button.setText("Play")
            else:
                self.animation.event_source.start()
                self.play_button.setText("Pause")
            self.playing = not self.playing

    def stop_animation(self):
        if self.animation:
            self.animation.event_source.stop()
            self.playing = False

    def deleteLater(self):
        self.stop_animation()
        super().deleteLater()


def compute_rectangle_given_center(X, Y, width, height, angle):
    """
    Calculate the coordinates of the vertices of a rectangle centered at (X, Y),
    with a given width, height, and rotated by 'angle' radians.

    Parameters:
    X (float): X-coordinate of the center point.
    Y (float): Y-coordinate of the center point.
    width (float): The width of the rectangle.
    height (float): The height of the rectangle.
    angle (float): The rotation angle in radians.

    Returns:
    list: A list of tuples containing the coordinates of the rectangle's vertices.
    """
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    half_height = height / 2
    front_X = X + half_height * cos_angle
    front_Y = Y + half_height * sin_angle

    rear_X = X - half_height * cos_angle
    rear_Y = Y - half_height * sin_angle

    half_width = width / 2
    front_left_X = front_X - half_width * sin_angle
    front_left_Y = front_Y + half_width * cos_angle
    front_right_X = front_X + half_width * sin_angle
    front_right_Y = front_Y - half_width * cos_angle

    rear_left_X = rear_X - half_width * sin_angle
    rear_left_Y = rear_Y + half_width * cos_angle
    rear_right_X = rear_X + half_width * sin_angle
    rear_right_Y = rear_Y - half_width * cos_angle

    return [
        (front_left_X, front_left_Y),
        (front_right_X, front_right_Y),
        (rear_right_X, rear_right_Y),
        (rear_left_X, rear_left_Y),
    ]


def calculate_ackerman_angles(steering_angle, wheelbase, width):
    """
    Calculate the left and right wheel steering angles for an Ackerman steering model.

    """
    if np.isclose(steering_angle, 0):
        return 0.0, 0.0

    # Calculate the turning radius from the bicycle model
    R = wheelbase / np.tan(steering_angle)

    R_L = R - (width / 2)
    R_R = R + (width / 2)

    steering_L = np.arctan2(wheelbase, R_L)
    steering_R = np.arctan2(wheelbase, R_R)

    return steering_L, steering_R


def draw_vehicle(
        current_state: State, steering_angle: float, vp: VehicleParams
) -> tuple[list[tuple[float, float]], list[list[tuple[float, float]]]]:
    vehicle_shape = compute_rectangle_given_center(
        current_state.X,
        current_state.Y,
        vp.width,
        vp.wheelbase,
        current_state.Psi,
    )
    front_axle_X, front_axle_Y = vp.front_axle_position(current_state)
    back_axle_X, back_axle_Y = vp.rear_axle_position(current_state)
    if vp.is_bicycle():
        front_wheel_shape = compute_rectangle_given_center(
            front_axle_X,
            front_axle_Y,
            vp.wheel_width,
            vp.wheel_diameter,
            current_state.Psi + steering_angle,
        )

        back_wheel_shape = compute_rectangle_given_center(
            back_axle_X,
            back_axle_Y,
            vp.wheel_width,
            vp.wheel_diameter,
            current_state.Psi,
        )

        return vehicle_shape, [front_wheel_shape, back_wheel_shape]
    else:
        front_left_vehicle = vehicle_shape[0]
        front_right_vehicle = vehicle_shape[1]

        (
            left_wheel_steering_angle,
            right_wheel_steering_angle,
        ) = calculate_ackerman_angles(steering_angle, vp.wheelbase, vp.width)

        front_left_wheel_shape = compute_rectangle_given_center(
            front_left_vehicle[0],
            front_left_vehicle[1],
            vp.wheel_width,
            vp.wheel_diameter,
            current_state.Psi + left_wheel_steering_angle,
        )

        front_right_wheel_shape = compute_rectangle_given_center(
            front_right_vehicle[0],
            front_right_vehicle[1],
            vp.wheel_width,
            vp.wheel_diameter,
            current_state.Psi + right_wheel_steering_angle,
        )

        back_left_vehicle = vehicle_shape[3]
        back_right_vehicle = vehicle_shape[2]

        back_left_wheel_shape = compute_rectangle_given_center(
            back_left_vehicle[0],
            back_left_vehicle[1],
            vp.wheel_width,
            vp.wheel_diameter,
            current_state.Psi,
        )

        back_right_wheel_shape = compute_rectangle_given_center(
            back_right_vehicle[0],
            back_right_vehicle[1],
            vp.wheel_width,
            vp.wheel_diameter,
            current_state.Psi,
        )

        return vehicle_shape, [
            front_left_wheel_shape,
            front_right_wheel_shape,
            back_left_wheel_shape,
            back_right_wheel_shape,
        ]


def draw_visual_range(current_state: State, vp: VehicleParams) -> list[tuple[float, float]]:
    front_axle_X, front_axle_Y = vp.front_axle_position(current_state)

    # Define ellipse parameters
    major_axis = 10
    minor_axis = 5
    num_points = 100  # The number of points to approximate the ellipse shape

    # Parametric equation for an ellipse
    t = np.linspace(0, 2 * np.pi, num_points)
    ellipse_x = major_axis * np.cos(t) / 2  # Divide by 2 since axes are diameters
    ellipse_y = minor_axis * np.sin(t) / 2

    # Rotate ellipse to align with the vehicle's Psi (heading angle)
    cos_psi = np.cos(current_state.Psi)
    sin_psi = np.sin(current_state.Psi)

    rotated_ellipse = [
        (front_axle_X + cos_psi * x - sin_psi * y, front_axle_Y + sin_psi * x + cos_psi * y)
        for x, y in zip(ellipse_x, ellipse_y)
    ]

    return rotated_ellipse
