from PyQt5.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QPushButton,
    QHBoxLayout,
)
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from dto.color_theme import ColorTheme


class ErrorPlotterWidget(QWidget):
    def __init__(
        self,
        color_theme: ColorTheme,
        title: str,
        trajectories,
        names,
        y_axis_label,
        x_axis_label="Time [s]",
        y_upper_bound=None,
        y_lower_bound=None,
        parent=None,
    ):
        super().__init__(parent=parent)
        self.setStyleSheet(
            f"background-color: {color_theme.primary_color}; color: {color_theme.text_color}"
        )
        plt.style.use(
            {
                "axes.facecolor": (1, 1, 1),
                "figure.facecolor": (0.9, 0.9, 0.9),
                "axes.edgecolor": "black",
                "axes.labelcolor": "black",
                "text.color": "black",
                "xtick.color": "black",
                "ytick.color": "black",
            }
        )

        self.title = title
        self.trajectories = trajectories
        self.highlighted = [False] * len(trajectories)
        self.visible = [True] * len(trajectories)
        self.colors = [
            "blue",
            "orange",
            "purple",
        ]
        self.names = names
        self.y_axis_label = y_axis_label
        self.x_axis_label = x_axis_label
        self.y_upper_bound = y_upper_bound
        self.y_lower_bound = y_lower_bound
        self.initUI()
        self.update_plots()

    def initUI(self):
        self.layout = QVBoxLayout(self)
        self.plot_canvas = PlotCanvas(
            title=self.title,
            names=self.names,
            colors=self.colors,
            y_axis_label=self.y_axis_label,
            x_axis_label=self.x_axis_label,
            y_upper_bound=self.y_upper_bound,
            y_lower_bound=self.y_lower_bound,
            width=5,
            height=4,
            parent=self,
        )
        self.toolbar = NavigationToolbar(self.plot_canvas, self)

        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.plot_canvas)

        # Layout for visibility toggle buttons
        self.visibility_layout = QHBoxLayout()
        self.layout.addLayout(self.visibility_layout)

        # Layout for highlight toggle buttons
        # self.highlight_layout = QHBoxLayout()
        # self.layout.addLayout(self.highlight_layout)
        #
        # for i, name in enumerate(self.names):
        #     vis_btn = QPushButton(f"Show {name}")
        #     vis_btn.setCheckable(True)
        #     vis_btn.setChecked(True)
        #     vis_btn.clicked.connect(
        #         lambda checked, index=i: self.toggle_visibility(index)
        #     )
        #     self.visibility_layout.addWidget(vis_btn)
        #
        #     highlight_btn = QPushButton(f"Highlight {name}")
        #     highlight_btn.setCheckable(True)
        #     highlight_btn.setChecked(False)
        #     highlight_btn.clicked.connect(
        #         lambda checked, index=i: self.toggle_highlight(index)
        #     )
        #     self.highlight_layout.addWidget(highlight_btn)

    def toggle_visibility(self, index):
        """Toggle the visibility of the selected trajectory."""
        self.visible[index] = not self.visible[index]
        sender = self.sender()
        sender.setText(
            f"{'Hide' if self.visible[index] else 'Show'} {self.names[index]}"
        )
        self.update_plots()

    def toggle_highlight(self, index):
        """Toggle the highlight state of the selected trajectory."""
        self.highlighted[index] = not self.highlighted[index]
        sender = self.sender()
        sender.setText(
            f"Unhighlight {self.names[index]}"
            if self.highlighted[index]
            else f"Highlight {self.names[index]}"
        )
        self.update_plots()

    def update_plots(self):
        """Update the plot based on current states of visibility and highlights."""
        self.plot_canvas.plot(
            self.trajectories,
            self.visible,
            self.highlighted,
        )


class PlotCanvas(FigureCanvas):
    def __init__(
        self,
        title: str,
        names: list[str],
        colors: list[str],
        y_axis_label: str,
        x_axis_label: str = "Time [s]",
        y_upper_bound=None,
        y_lower_bound=None,
        width=5,
        height=4,
        dpi=100,
        parent=None,
    ):
        self.fig = Figure(figsize=(width, height), dpi=dpi)

        self.axes = self.fig.add_subplot(111)
        self.title = title
        self.names = names
        self.colors = colors
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.y_upper_bound = y_upper_bound
        self.y_lower_bound = y_lower_bound

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        self.initial_draw = True
        self.initial_plot()

    def initial_plot(self):
        """Initialize an empty plot or a placeholder plot."""
        self.axes.clear()
        self.axes.set_title(self.title)
        self.axes.set_xlabel(self.x_axis_label)
        self.axes.set_ylabel(self.y_axis_label)
        self.axes.plot([], [], "r-")  # You can adjust this for a better initial view
        self.draw()

    def plot(self, trajectories, visible, highlighted):
        """Plot all trajectories based on visibility and highlight status using specific colors."""
        # Save the current zoom level and position
        if self.initial_draw:
            self.initial_draw = False
        else:
            xlim, ylim = (
                self.axes.get_xlim(),
                self.axes.get_ylim(),
            )  # Store the current limits

            self.axes.clear()
            self.axes.set_title(self.title)
            self.axes.set_xlabel(self.x_axis_label)
            self.axes.set_ylabel(self.y_axis_label)

            self.axes.set_xlim(xlim)  # Restore the limits
            self.axes.set_ylim(ylim)

        for i, (times, values) in enumerate(trajectories):
            if visible[i]:
                # Show the mean value of the trajectory on the label
                mean_value = sum(values) / len(values)
                label = (
                    f"{self.names[i]} (avg: {mean_value:.2f} {self.y_axis_label})"
                    if self.names
                    else f"Trajectory {i + 1} (avg: {mean_value:.2f} {self.y_axis_label})"
                )
                color = self.colors[i % len(self.colors)]
                if highlighted[i]:
                    self.axes.plot(
                        times,
                        values,
                        marker="o",
                        markersize=5,
                        label=label,
                        linewidth=3,
                        color=color,
                    )
                else:
                    self.axes.plot(
                        times,
                        values,
                        marker="o",
                        markersize=3,
                        label=label,
                        linewidth=1,
                        color=color,
                    )

        if self.y_upper_bound is not None:
            self.axes.axhline(
                y=self.y_upper_bound,
                color="red",
                linestyle="--",
                label="Upper Bound",
            )

        if self.y_lower_bound is not None:
            self.axes.axhline(
                y=self.y_lower_bound,
                color="red",
                linestyle="--",
                label="Lower Bound",
            )

        self.axes.legend()

        self.draw()