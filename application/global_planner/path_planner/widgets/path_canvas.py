from enum import Enum, auto

import numpy as np
import pandas as pd
from PyQt5.QtCore import pyqtSignal, QPointF, Qt, QRectF
from PyQt5.QtGui import QPen, QColor, QPainter
from PyQt5.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsEllipseItem,
    QGraphicsTextItem,
)
from numpy import ndarray

from application.core.graphics.composite_item import CompositeItem
from application.core.graphics.draggable_rect import DraggableRect
from dto.color_theme import ColorTheme
from dto.geometry import Rectangle
from dto.waypoint import WaypointWithHeading, Waypoint


class DrawMode(Enum):
    Only_Select = auto()  # Only select items
    Add_Waypoint = auto()  # Add waypoints or select items
    Add_Obstacle = auto()  # Add obstacles or select items


class PathCanvas(QGraphicsView):
    pathParamsChanged = pyqtSignal(tuple, name="pathParamsChanged")

    def __init__(
        self,
        world_x_limit: float,
        world_y_limit: float,
        color_theme: ColorTheme,
        parent=None,
    ):
        super().__init__(parent)
        self.last_piece_wise_spiral_dfs = None
        self.mode = DrawMode.Add_Waypoint
        self.color_theme = color_theme
        self.world_x_limit = world_x_limit
        self.world_y_limit = world_y_limit

        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self.setRenderHint(QPainter.Antialiasing)
        self.setBackgroundBrush(QColor(self.color_theme.secondary_color))
        self.setMouseTracking(True)

        self.mouse_pos_text_item = QGraphicsTextItem("")
        self.scene.addItem(self.mouse_pos_text_item)
        self.mouse_pos_text_item.setDefaultTextColor(Qt.black)
        self.mouse_pos_text_item.setZValue(100)  # Ensure the text item is always on top

        self.waypoints_with_headings: list[WaypointWithHeading] = []
        self.composite_path_items: list[CompositeItem] = []
        self.obstacles: list[DraggableRect] = []
        self.bounds: dict[
            (WaypointWithHeading, WaypointWithHeading), [QGraphicsEllipseItem]
        ] = {}

        self.selected_item = None
        self.start_pos_of_new_rectangle = None
        self.temp_rect = None
        self.drag_start_pos = None
        self.drag_threshold = 20  # threshold before a click is considered a drag

    @property
    def world_to_canvas_x_ratio(self):
        """Ratio of world coordinates (meters) to canvas coordinates (pixels) in the x-direction."""
        return self.viewport().width() / self.world_x_limit

    @property
    def world_to_canvas_y_ratio(self):
        return self.viewport().height() / self.world_y_limit

    @property
    def canvas_to_world_x_ratio(self):
        return self.world_x_limit / self.viewport().width()

    @property
    def canvas_to_world_y_ratio(self):
        return self.world_y_limit / self.viewport().height()

    def recalculate_positions(self, new_world_x_limit: float, new_world_y_limit: float):
        """Recalculates positions and sizes of all elements in the scene."""
        self.setSceneRect(0, 0, self.viewport().width(), self.viewport().height())

        # Recalculate waypoints and obstacles positions
        old_rect_path_frame = []
        for obstacle in self.obstacles:
            rect = obstacle.get_scene_rect()
            path_rect = self.canvas_to_world_rectangle(rect)
            old_rect_path_frame.append(path_rect)
        self.obstacles = []

        self.world_x_limit = new_world_x_limit
        self.world_y_limit = new_world_y_limit
        self.clear_all()

        for wp in self.waypoints_with_headings:
            self.add_composite_item(
                self.world_to_canvas_waypoint(wp),
                initial_angle=wp.heading,
                emit_changes=False,
            )
        for obstacle, old_path_rect in zip(self.obstacles, old_rect_path_frame):
            new_scene_rect = self.world_to_canvas_rectangle(old_path_rect)
            obstacle = self.add_obstacle(
                new_scene_rect.x(),
                new_scene_rect.y(),
                new_scene_rect.width(),
                new_scene_rect.height(),
            )
            self.obstacles.append(obstacle)

        self.update_bounds(self.last_piece_wise_spiral_dfs, all=True)

    def clear_all(self):
        self.scene.clear()
        self.mouse_pos_text_item = QGraphicsTextItem("")
        self.scene.addItem(self.mouse_pos_text_item)
        self.mouse_pos_text_item.setDefaultTextColor(Qt.black)
        self.mouse_pos_text_item.setZValue(100)  # Ensure the text item is always on top

        self.composite_path_items = []
        self.bounds = {}
        self.obstacles = []
        self.selected_item = None
        self.start_pos_of_new_rectangle = None
        self.temp_rect = None

    def init_path(
        self,
        waypoints_with_headings: list[WaypointWithHeading],
        obstacles: list[Rectangle] = None,
    ):
        self.clear_all()
        if obstacles is None:
            obstacles = []
        for obstacle in obstacles:
            scene_rect = self.world_to_canvas_rectangle(obstacle)
            obst = self.add_obstacle(
                scene_rect.x(), scene_rect.y(), scene_rect.width(), scene_rect.height()
            )
            self.obstacles.append(obst)

        for wp in waypoints_with_headings:
            self.add_composite_item(
                self.world_to_canvas_waypoint(wp),
                initial_angle=wp.heading,
                emit_changes=False,
            )
        self.waypoints_with_headings = waypoints_with_headings
        self.pathParamsChanged.emit((waypoints_with_headings, obstacles))

    def world_to_canvas(self, X, Y):
        X = X * self.world_to_canvas_x_ratio
        Y = (self.world_y_limit - Y) * self.world_to_canvas_y_ratio
        return X, Y

    def world_to_canvas_waypoint(self, waypoint: Waypoint):
        return QPointF(
            waypoint.x * self.world_to_canvas_x_ratio,
            (self.world_y_limit - waypoint.y) * self.world_to_canvas_y_ratio,
        )

    def canvas_to_world_waypoint(self, point: QPointF):
        return Waypoint(
            point.x() * self.canvas_to_world_x_ratio,
            self.world_y_limit - point.y() * self.canvas_to_world_y_ratio,
        )

    def canvas_to_world_waypoint_w_heading(self, point: QPointF, heading: float):
        return WaypointWithHeading(
            point.x() * self.canvas_to_world_x_ratio,
            self.world_y_limit - point.y() * self.canvas_to_world_y_ratio,
            heading,
        )

    def canvas_to_world_rectangle(self, scene_rect: QRectF):
        """
        Convert a rectangle from draw canvas coordinates to world (inertial) coordinates.
        Adjusts for the top-left (canvas) -> bottom-left origin of the inertial frame.

        @param scene_rect: QRectF with top-left corner (x, y) and dimensions (width, height) in canvas coordinates.
        @return: Rectangle in inertial coordinates, where the origin is at the bottom-left.
        """
        path_rect_x = scene_rect.x() * self.canvas_to_world_x_ratio
        path_rect_y = (
            self.world_y_limit
            - (scene_rect.y() + scene_rect.height()) * self.canvas_to_world_y_ratio
        )
        path_rect_width = scene_rect.width() * self.canvas_to_world_x_ratio
        path_rect_height = scene_rect.height() * self.canvas_to_world_y_ratio

        return Rectangle(path_rect_x, path_rect_y, path_rect_width, path_rect_height)

    def world_to_canvas_rectangle(self, path_rect: Rectangle):
        """
        Convert a rectangle from world (inertial) coordinates to draw canvas coordinates.
        Adjusts for the bottom-left (inertial) -> top-left origin of the canvas frame.

        @param path_rect: Rectangle in inertial coordinates, where the origin is at the bottom-left.
        @return: QRectF with top-left corner (x, y) and dimensions (width, height) in canvas coordinates.
        """
        scene_rect_x = path_rect.x * self.world_to_canvas_x_ratio
        scene_rect_y = (
            self.world_y_limit - (path_rect.y + path_rect.height)
        ) * self.world_to_canvas_y_ratio
        scene_rect_width = path_rect.width * self.world_to_canvas_x_ratio
        scene_rect_height = path_rect.height * self.world_to_canvas_y_ratio
        return QRectF(scene_rect_x, scene_rect_y, scene_rect_width, scene_rect_height)

    def draw_points(self, X: ndarray, Y: ndarray):
        # Prepare the pen and brush only once for all ellipses
        pen = QPen(QColor("blue"))
        brush = QColor(self.color_theme.primary_color)

        points = []
        circle_radius = 2
        self.setUpdatesEnabled(False)
        for x, y in zip(X, Y):
            ellipse = self.scene.addEllipse(
                x - circle_radius / 2,  # Center of the circle
                y - circle_radius / 2,
                circle_radius,  # Width of the circle
                circle_radius,  # Height of the circle
                pen,
                brush,
            )
            points.append(ellipse)
        self.setUpdatesEnabled(True)
        return points

    def update_bounds(
        self, piece_wise_spiral_dfs: list[pd.DataFrame] = None, all=False
    ):
        if piece_wise_spiral_dfs is None:
            if self.last_piece_wise_spiral_dfs is None:
                print("No piece-wise spiral dataframes to update bounds")
                return
            print("Using last piece-wise spiral dataframes")
        else:
            self.last_piece_wise_spiral_dfs = piece_wise_spiral_dfs

        piece_wise_X_bounds = [
            np.concatenate((df.X_bounds_positive, df.X_bounds_negative))
            for df in self.last_piece_wise_spiral_dfs
        ]
        piece_wise_Y_bounds = [
            np.concatenate((df.Y_bounds_positive, df.Y_bounds_negative))
            for df in self.last_piece_wise_spiral_dfs
        ]

        if all:
            for ellipse_items in self.bounds.values():
                for ellipse_item in ellipse_items:
                    self.scene.removeItem(ellipse_item)
            self.bounds = {}

        if self.bounds == {}:
            for i in range(len(self.waypoints_with_headings) - 1):
                wp = self.waypoints_with_headings[i]
                wp_next = self.waypoints_with_headings[i + 1]
                X_bounds = piece_wise_X_bounds[i]
                Y_bounds = piece_wise_Y_bounds[i]

                X_transformed, Y_transformed = self.world_to_canvas(X_bounds, Y_bounds)
                self.bounds[(wp, wp_next)] = self.draw_points(
                    X_transformed, Y_transformed
                )
        else:
            new_bounds = {}
            for i in range(len(self.waypoints_with_headings) - 1):
                wp = self.waypoints_with_headings[i]
                wp_next = self.waypoints_with_headings[i + 1]
                if (wp, wp_next) not in self.bounds:
                    X_bounds = piece_wise_X_bounds[i]
                    Y_bounds = piece_wise_Y_bounds[i]

                    X_transformed, Y_transformed = self.world_to_canvas(
                        X_bounds, Y_bounds
                    )
                    new_bounds[(wp, wp_next)] = self.draw_points(
                        X_transformed, Y_transformed
                    )
                else:
                    new_bounds[(wp, wp_next)] = self.bounds[(wp, wp_next)]

            for (wp, wp_next), ellipse_items in self.bounds.items():
                if (wp, wp_next) not in new_bounds:
                    for ellipse_item in ellipse_items:
                        self.scene.removeItem(ellipse_item)
            self.bounds = new_bounds

    def emit_new_path_params(self):
        self.waypoints_with_headings = [
            self.canvas_to_world_waypoint_w_heading(ci.pos(), ci.angle_radians)
            for ci in self.composite_path_items
        ]

        obstacles = [
            self.canvas_to_world_rectangle(obstacle.get_scene_rect())
            for obstacle in self.obstacles
        ]

        self.pathParamsChanged.emit((self.waypoints_with_headings, obstacles))

    def set_itm_as_selected(self, item):
        if self.selected_item and self.selected_item != item:
            self.selected_item.setSelected(False)
        self.selected_item = item
        if self.selected_item:
            self.selected_item.setSelected(True)

    def add_composite_item(self, click_point, initial_angle=0, emit_changes=True):
        ci = CompositeItem(initial_rotation_radians=initial_angle)
        ci.setPos(click_point)
        ci.valueChanged.connect(self.emit_new_path_params)

        self.scene.addItem(ci)
        self.composite_path_items.append(ci)

        self.set_itm_as_selected(ci)
        if emit_changes:
            self.emit_new_path_params()

    def add_obstacle(self, x, y, width, height, color=QColor("black")):
        rect = DraggableRect(x, y, width, height, color)
        rect.valueChanged.connect(self.emit_new_path_params)
        self.scene.addItem(rect)

        return rect

    def mousePressEvent(self, event):
        super(PathCanvas, self).mousePressEvent(event)

        self.set_itm_as_selected(None)  # Deselect any selected item

        if self.temp_rect:
            self.scene.removeItem(self.temp_rect)
            self.temp_rect = None
            self.start_pos_of_new_rectangle = None

        if event.button() == Qt.LeftButton:
            self.drag_start_pos = event.pos()
            click_point = self.mapToScene(event.pos())

            items = self.scene.items(click_point)
            if items:
                for item in items:
                    if isinstance(item, CompositeItem) or isinstance(
                        item, DraggableRect
                    ):
                        self.set_itm_as_selected(item)
                        break  # Only select the first item
            else:
                if self.mode == DrawMode.Add_Waypoint:
                    self.add_composite_item(click_point)
                elif self.mode == DrawMode.Add_Obstacle:
                    self.start_pos_of_new_rectangle = click_point

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)

        click_point = self.mapToScene(event.pos())
        world_click_point = self.canvas_to_world_waypoint(click_point)
        self.mouse_pos_text_item.setPlainText(
            f"({click_point.x():.2f}, {click_point.y():.2f}, {world_click_point.x:.2f}, {world_click_point.y:.2f})"
        )
        self.adjustMouseTextPosition(click_point)

        if event.buttons() & Qt.LeftButton and self.drag_start_pos is not None:
            drag_distance = event.pos() - self.drag_start_pos
            if drag_distance.manhattanLength() > self.drag_threshold:
                # We are dragging
                if (
                    self.mode == DrawMode.Add_Obstacle
                    and self.start_pos_of_new_rectangle
                ):
                    end_point = self.mapToScene(event.pos())

                    if self.temp_rect:
                        self.scene.removeItem(self.temp_rect)

                    width = end_point.x() - self.start_pos_of_new_rectangle.x()
                    height = end_point.y() - self.start_pos_of_new_rectangle.y()

                    # Always draw from the top left corner
                    x = min(self.start_pos_of_new_rectangle.x(), end_point.x())
                    y = min(self.start_pos_of_new_rectangle.y(), end_point.y())

                    self.temp_rect = self.add_obstacle(x, y, abs(width), abs(height))

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)

        if self.temp_rect:
            if self.mode == DrawMode.Add_Obstacle:
                self.obstacles.append(self.temp_rect)
                self.emit_new_path_params()
            else:
                self.scene.removeItem(self.temp_rect)
            self.temp_rect = None
            self.start_pos_of_new_rectangle = None
        self.drag_start_pos = None

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        if self.selected_item and event.key() in (Qt.Key_Backspace, Qt.Key_Delete):
            if isinstance(self.selected_item, CompositeItem):
                self.composite_path_items.remove(self.selected_item)
            elif isinstance(self.selected_item, DraggableRect):
                self.obstacles.remove(self.selected_item)
            if self.selected_item in self.scene.items():
                self.scene.removeItem(self.selected_item)
            self.selected_item = None
            self.emit_new_path_params()
        else:
            print("No item selected to delete")

    def wheelEvent(self, event):
        if self.selected_item and isinstance(self.selected_item, CompositeItem):
            delta_angle_degrees = event.angleDelta().y() / 120
            self.selected_item.rotate(delta_angle_degrees)

    def resizeEvent(self, event):
        self.setSceneRect(0, 0, self.viewport().width(), self.viewport().height())
        super().resizeEvent(event)

    def adjustMouseTextPosition(self, pos):
        offset_x = 15  # Horizontal offset for the text position
        offset_y = 15  # Vertical offset for the text position

        # Get the dimensions of the view
        view_width = self.viewport().width()
        view_height = self.viewport().height()

        # Calculate scene coordinates for the center of the view
        center_x = view_width / 2
        center_y = view_height / 2

        # Determine the quadrant and adjust the text position accordingly
        if pos.x() < center_x and pos.y() < center_y:  # Top-Left
            self.mouse_pos_text_item.setPos(pos.x() + offset_x, pos.y() + offset_y)
        elif pos.x() >= center_x and pos.y() < center_y:  # Top-Right
            self.mouse_pos_text_item.setPos(
                pos.x() - self.mouse_pos_text_item.boundingRect().width() - offset_x,
                pos.y() + offset_y,
            )
        elif pos.x() < center_x and pos.y() >= center_y:  # Bottom-Left
            self.mouse_pos_text_item.setPos(
                pos.x() + offset_x,
                pos.y() - self.mouse_pos_text_item.boundingRect().height() - offset_y,
            )
        else:  # Bottom-Right
            self.mouse_pos_text_item.setPos(
                pos.x() - self.mouse_pos_text_item.boundingRect().width() - offset_x,
                pos.y() - self.mouse_pos_text_item.boundingRect().height() - offset_y,
            )