import networkx as nx
from PyQt5 import QtCore
from PyQt5.QtGui import QIcon, QCursor
from PyQt5.QtWidgets import QAction
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from dto.color_theme import ColorTheme
from global_planner.global_planner import GlobalPlanner
from global_planner.road_network import RoadNetwork


class CustomToolbar(NavigationToolbar2QT):
    def __init__(self, canvas, parent, coordinates=True):
        super().__init__(canvas, parent, coordinates)
        self.coordinates = False  # Disable the coordinates display

        # Add a toggle button for dragging nodes
        self.toggle_drag_action = QAction(QIcon(), "Toggle Node Dragging", self)
        self.toggle_drag_action.setCheckable(True)
        self.toggle_drag_action.setChecked(False)
        self.toggle_drag_action.triggered.connect(self.toggle_dragging)
        self.addAction(self.toggle_drag_action)

        self.node_dragging_enabled = False

    def toggle_dragging(self, checked):
        self.node_dragging_enabled = checked
        self.canvas.setCursor(
            QCursor(QtCore.Qt.PointingHandCursor if checked else QtCore.Qt.ArrowCursor)
        )


class RoadNetworkCanvas(FigureCanvas):
    path_nodes_Changed = QtCore.pyqtSignal(list)

    def __init__(
        self,
        road_network_name: str,
        road_network: RoadNetwork,
        color_theme: ColorTheme,
        visualize_only=False,
        initial_path: list[str] = None,
    ):
        figsize = (4, 4) if visualize_only else (12, 12)
        self.fig, self.ax = plt.subplots(figsize=figsize)
        super(RoadNetworkCanvas, self).__init__(self.fig)

        self.road_network_name = road_network_name
        self.road_network = road_network
        self.color_theme = color_theme
        self.visualize_only = visualize_only

        if visualize_only:
            self.fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
            self.selected_node_ids = []
            self.path_node_ids = []
        else:
            self.fig.subplots_adjust(left=0.01, bottom=0.03, right=0.95, top=0.95)
            self.toolbar = CustomToolbar(self, self)

            self.mpl_connect("button_press_event", self.on_click)
            self.mpl_connect("button_release_event", self.on_release)
            self.mpl_connect("motion_notify_event", self.on_motion)
            self.last_clicked_node = None

            self.global_planner = GlobalPlanner()
            if initial_path:
                self.selected_node_ids = initial_path
                self.path_node_ids = initial_path
            else:
                self.selected_node_ids = []
                self.path_node_ids = []

        self.draw_graph(initial_draw=True)

    def draw_graph(self, initial_draw=False):
        if not initial_draw:
            xlim, ylim = (
                self.ax.get_xlim(),
                self.ax.get_ylim(),
            )  # Store the current limits

            self.ax.clear()

            self.ax.set_xlim(xlim)  # Restore the limits
            self.ax.set_ylim(ylim)

        pos = {
            node: (data["x"], data["y"])
            for node, data in self.road_network.nodes(data=True)
        }

        node_colors = [
            "green" if node in self.selected_node_ids else "black"
            for node in self.road_network.nodes()
        ]

        if self.path_node_ids and len(self.path_node_ids) > 1:
            edge_colors = [
                "red"
                if (str(u), str(v)) in zip(self.path_node_ids, self.path_node_ids[1:])
                else "blue"
                for u, v in self.road_network.edges()
            ]
        else:
            edge_colors = ["blue" for _ in self.road_network.edges()]

        nx.draw_networkx(
            self.road_network.G,
            width=2,
            pos=pos,
            ax=self.ax,
            arrows=True,
            arrowsize=10,
            with_labels=True,
            node_size=350,
            node_color=node_colors,
            font_size=6,
            font_color="lightblue",
            edge_color=edge_colors,
        )

        self.ax.set_title(self.road_network_name)
        self.ax.axis("on")
        if not self.visualize_only:
            self.ax.set_xlabel("X Coordinate")
            self.ax.set_ylabel("Y Coordinate")
            self.ax.yaxis.set_label_position("right")

        self.draw()

    def on_click(self, event):
        if self.visualize_only:
            return

        closest_node = self.road_network.get_closest_node(
            event.xdata, event.ydata, threshold=0.1
        )
        if closest_node:
            closest_node_id, _, _ = closest_node

            if self.toolbar.node_dragging_enabled:
                self.last_clicked_node = closest_node_id
            else:
                self.handle_node_selection(closest_node_id)

    def handle_node_selection(self, nearest_node):
        if nearest_node in self.selected_node_ids:
            self.selected_node_ids.remove(nearest_node)
        else:
            self.selected_node_ids.append(nearest_node)

        self.path_node_ids = self.global_planner.find_shortest_path_with_stops(
            self.road_network, self.selected_node_ids
        )
        self.path_nodes_Changed.emit(self.path_node_ids)
        self.draw_graph()

    def on_motion(self, event):
        if (
            not self.visualize_only
            and self.last_clicked_node
            and self.toolbar.node_dragging_enabled
            and event.xdata
            and event.ydata
        ):
            # Update the position of the node being dragged
            self.road_network.update_node_position(
                self.last_clicked_node, event.xdata, event.ydata
            )
            self.path_nodes_Changed.emit(
                self.path_node_ids
            )  # Update the path, since one of the nodes has moved
            self.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
            self.draw_graph()

    def on_release(self, event):
        if self.toolbar.node_dragging_enabled and not self.visualize_only:
            self.last_clicked_node = None  # Release the node
            self.draw_graph()