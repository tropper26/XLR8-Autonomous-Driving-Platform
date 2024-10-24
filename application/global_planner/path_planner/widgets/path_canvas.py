from enum import Enum, auto

from PyQt5.QtCore import pyqtSignal, QPointF, Qt, QRectF
from PyQt5.QtGui import QPen, QColor, QPainter
from PyQt5.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsEllipseItem,
    QGraphicsTextItem,
)
from numpy import ndarray

from application.application_status import ApplicationStatus
from application.core.graphics.composite_item import WaypointCompositeItem
from application.core.graphics.draggable_rect import DraggableRect
from application.core.zoomable_graphics_view import ZoomableGraphicsView
from dto.color_theme import ColorTheme
from dto.geometry import Rectangle
from dto.waypoint import WaypointWithHeading, Waypoint
from global_planner.path_planning.pathmanager import PathManager
from parametric_curves.path_segment import PathSegment


class Mode(Enum):
    Only_Select = auto()  # Only select items
    Add_Waypoint = auto()  # Add waypoints or select items
    Insert_Waypoint = auto()  # Insert waypoints or select items
    Add_Obstacle = auto()  # Add obstacles or select items
    Zoom = auto()  # Zoom in/out


class PathCanvas(ZoomableGraphicsView):
    obstaclesChanged = pyqtSignal(list, name="obstaclesChanged")

    def __init__(
        self,
        current_app_status: ApplicationStatus,
        world_x_limit: float,
        world_y_limit: float,
        color_theme: ColorTheme,
        parent=None,
    ):
        super().__init__(parent)
        self.mode = Mode.Add_Waypoint
        self.color_theme = color_theme
        self.world_x_limit = world_x_limit
        self.world_y_limit = world_y_limit
        self.current_app_status = current_app_status

        self.path_manager = PathManager(
            left_lane_count=1,
            right_lane_count=1,
            lane_widths=[
                self.current_app_status.lane_width
                for _ in range(self.current_app_status.lane_count + 2 + 1)
            ],
        )  # TODO update UI to be able to set lane widths

        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self.setRenderHint(QPainter.Antialiasing)
        self.setBackgroundBrush(QColor(self.color_theme.secondary_color))
        self.setMouseTracking(True)

        self.mouse_pos_text_item = QGraphicsTextItem("")
        self.scene.addItem(self.mouse_pos_text_item)
        self.mouse_pos_text_item.setDefaultTextColor(Qt.black)
        self.mouse_pos_text_item.setZValue(100)  # Ensure the text item is always on top

        self.route_waypoint_objects: list[WaypointCompositeItem] = []
        self.obstacles: list[DraggableRect] = []
        self.discretized_segments: list[list[QGraphicsEllipseItem]] = []
        self.selected_item = None
        self.start_pos_of_new_rectangle = None
        self.temp_rect = None
        self.drag_start_pos = None
        self.drag_threshold = 20  # pixel threshold before a click is considered a drag

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

    def recalculate_positions(
        self,
        new_world_x_limit: float,
        new_world_y_limit: float,
    ):
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
        self.clear_and_reset_all(clear_path_manager=False)

        for wp in self.path_manager.route:
            self.append_waypoint(
                self.world_to_canvas_waypoint(wp),
                initial_angle=wp.heading,
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

    def clear_screen(self):
        self.scene.clear()
        self.mouse_pos_text_item = QGraphicsTextItem("")
        self.scene.addItem(self.mouse_pos_text_item)
        self.mouse_pos_text_item.setDefaultTextColor(Qt.black)
        self.mouse_pos_text_item.setZValue(100)  # Ensure the text item is always on top

    def clear_and_reset_all(self, clear_path_manager: bool = True):
        self.clear_screen()

        if clear_path_manager:
            self.path_manager.clear()

        self.route_waypoint_objects = []
        self.discretized_segments = []
        self.obstacles = []
        self.selected_item = None
        self.start_pos_of_new_rectangle = None
        self.temp_rect = None

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

    def canvas_to_world_waypoint_w_heading(
        self, point: QPointF, heading: float, index_in_route: int = None
    ):
        return WaypointWithHeading(
            point.x() * self.canvas_to_world_x_ratio,
            self.world_y_limit - point.y() * self.canvas_to_world_y_ratio,
            heading,
            index_in_route=index_in_route,
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

    def set_itm_as_selected(self, item):
        if self.selected_item and self.selected_item != item:
            self.selected_item.setSelected(False)
        self.selected_item = item
        if self.selected_item:
            self.selected_item.setSelected(True)

    def init_path_by_route(
        self,
        route: list[WaypointWithHeading],
        obstacles: list[Rectangle] = None,
    ):
        self.clear_and_reset_all()
        if obstacles is None:
            obstacles = []
        for obstacle in obstacles:
            scene_rect = self.world_to_canvas_rectangle(obstacle)
            obst = self.add_obstacle(
                scene_rect.x(), scene_rect.y(), scene_rect.width(), scene_rect.height()
            )
            self.obstacles.append(obst)

        index = 0
        for wp in route:
            waypoint_object = self.draw_waypoint(
                index=index,
                position=self.world_to_canvas_waypoint(wp),
                initial_angle=wp.heading,
            )
            index += 1
            self.route_waypoint_objects.append(waypoint_object)

        path_segments = self.path_manager.create_path_segments_from_route(
            route=route,
            step_size=self.current_app_status.path_visualisation_step_size,
        )

        for path_segment in path_segments:
            self.discretized_segments.append(
                self.draw_discretized_segment(path_segment)
            )

    def update_waypoint(self, waypoint_object: WaypointCompositeItem):
        waypoint = self.canvas_to_world_waypoint_w_heading(
            waypoint_object.pos(),
            waypoint_object.angle_radians,
            waypoint_object.index_in_route,
        )

        updated_path_segments, updated_path_segments_indexes = (
            self.path_manager.update_waypoint(
                modified_route_waypoint=waypoint,
                step_size=self.current_app_status.path_visualisation_step_size,
            )
        )

        self.erase_segments_by_index(updated_path_segments_indexes)

        for updated_path_segment, updated_path_segment_index in zip(
            updated_path_segments, updated_path_segments_indexes
        ):
            self.discretized_segments[updated_path_segment_index] = (
                self.draw_discretized_segment(updated_path_segment)
            )

    def append_waypoint(self, position: QPointF, initial_angle: float = 0.0):
        waypoint_object = self.draw_waypoint(
            index=len(self.route_waypoint_objects),
            position=position,
            initial_angle=initial_angle,
        )
        self.route_waypoint_objects.append(waypoint_object)

        waypoint = self.canvas_to_world_waypoint_w_heading(
            waypoint_object.pos(),
            waypoint_object.angle_radians,
            waypoint_object.index_in_route,
        )

        if len(self.route_waypoint_objects) == 1:
            self.path_manager.add_first_waypoint(waypoint)
            return

        new_segment, new_segment_index = self.path_manager.append_waypoint(
            waypoint, step_size=self.current_app_status.path_visualisation_step_size
        )
        self.discretized_segments.append(self.draw_discretized_segment(new_segment))

    def insert_waypoint(
        self, position: QPointF, index: int, initial_angle: float = 0.0
    ):
        if index < 0 or index >= len(self.route_waypoint_objects):
            raise ValueError(
                f"Invalid index {index} for inserting waypoint. Must be between 0 and {len(self.route_waypoint_objects) - 1}"
            )

        waypoint_object = self.draw_waypoint(
            index=index,
            position=position,
            initial_angle=initial_angle,
        )

        self.route_waypoint_objects.insert(index, waypoint_object)
        # update index_in_route for all subsequent waypoints
        for i in range(index + 1, len(self.route_waypoint_objects)):
            self.route_waypoint_objects[i].index_in_route += 1

        waypoint = self.canvas_to_world_waypoint_w_heading(
            waypoint_object.pos(),
            waypoint_object.angle_radians,
            waypoint_object.index_in_route,
        )

        index = waypoint.index_in_route

        if index == 0:
            new_segment, new_segment_index = self.path_manager.insert_waypoint_start(
                waypoint, step_size=self.current_app_status.path_visualisation_step_size
            )
            self.discretized_segments.insert(
                0, self.draw_discretized_segment(new_segment)
            )
            return

        # Insert the waypoint in the middle of the route
        changed_path_segments, changed_path_segments_indexes = (
            self.path_manager.insert_waypoint(
                new_waypoint=waypoint,
                step_size=self.current_app_status.path_visualisation_step_size,
            )
        )

        self.erase_segment_by_index(index)
        self.discretized_segments.pop(index)

        for changed_path_segments, changed_path_segments_indexes in zip(
            changed_path_segments, changed_path_segments_indexes
        ):
            self.discretized_segments.insert(
                changed_path_segments_indexes,
                self.draw_discretized_segment(changed_path_segments),
            )

    def remove_waypoint(self, waypoint_object: WaypointCompositeItem):
        self.route_waypoint_objects.pop(waypoint_object.index_in_route)
        # update index_in_route for all subsequent waypoints
        for i in range(
            waypoint_object.index_in_route, len(self.route_waypoint_objects)
        ):
            self.route_waypoint_objects[i].index_in_route -= 1

        self.scene.removeItem(waypoint_object)

        index = waypoint_object.index_in_route

        if index == 0:
            self.path_manager.remove_first_waypoint()

            if len(self.discretized_segments) > 0:
                self.erase_segment_by_index(0)
                self.discretized_segments.pop(0)
            return

        if index == len(self.route_waypoint_objects):
            self.path_manager.remove_last_waypoint()
            self.erase_segment_by_index(-1)
            self.discretized_segments.pop(-1)
            return

        replacement_segment = self.path_manager.remove_waypoint_by_index(
            removed_waypoint_index=index,
            step_size=self.current_app_status.path_visualisation_step_size,
        )

        segment_as_points = self.draw_discretized_segment(replacement_segment)

        self.erase_segments_by_index([index - 1, index])
        # Replace the segment which came before the deleted waypoint
        self.discretized_segments[index - 1] = segment_as_points
        # Remove segment which came after the deleted waypoint
        self.discretized_segments.pop(index)

    def draw_waypoint(self, index: int, position: QPointF, initial_angle: float = 0.0):
        waypoint_object = WaypointCompositeItem(
            index_in_route=index,
            initial_rotation_radians=initial_angle,
        )
        waypoint_object.setPos(position)
        waypoint_object.valueChanged.connect(self.update_waypoint)

        self.scene.addItem(waypoint_object)

        self.set_itm_as_selected(waypoint_object)

        return waypoint_object

    def draw_discretized_segment(self, segment: PathSegment):
        # TODO: Implement curve selection
        curves_to_show = [True, False, True, False, True, False, True]  # list of booleans, which curves to show
        # Use boolean list to select the curves (columns) to show
        lateral_X_to_show = segment.discretized.lateral_X[:, curves_to_show].flatten()
        lateral_Y_to_show = segment.discretized.lateral_Y[:, curves_to_show].flatten()

        X_transformed, Y_transformed = self.world_to_canvas(lateral_X_to_show, lateral_Y_to_show)
        return self.draw_points(X_transformed, Y_transformed)

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

    def erase_segments_by_index(self, segment_indexes_to_remove: list[int]):
        self.setUpdatesEnabled(False)
        for segment_index in segment_indexes_to_remove:
            for ellipse_item in self.discretized_segments[segment_index]:
                self.scene.removeItem(ellipse_item)
        self.setUpdatesEnabled(True)

    def erase_segment_by_index(self, segment_index: int):
        self.setUpdatesEnabled(False)
        for ellipse_item in self.discretized_segments[segment_index]:
            self.scene.removeItem(ellipse_item)
        self.setUpdatesEnabled(True)

    def emit_new_obstacles(self):
        obstacles = [
            self.canvas_to_world_rectangle(obstacle.get_scene_rect())
            for obstacle in self.obstacles
        ]
        self.obstaclesChanged.emit(obstacles)

    def add_obstacle(self, x, y, width, height, color=QColor("black")):
        rect = DraggableRect(x, y, width, height, color)
        rect.valueChanged.connect(self.emit_new_obstacles)
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
                    if isinstance(item, WaypointCompositeItem) or isinstance(
                        item, DraggableRect
                    ):
                        self.set_itm_as_selected(item)
                        break  # Only select the first item
            else:
                if self.mode == Mode.Add_Waypoint:
                    self.append_waypoint(click_point)
                elif self.mode == Mode.Insert_Waypoint:
                    self.insert_waypoint(
                        click_point, index=0
                    )  # TODO: Implement choice of index
                elif self.mode == Mode.Add_Obstacle:
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
                if self.mode == Mode.Add_Obstacle and self.start_pos_of_new_rectangle:
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
            if self.mode == Mode.Add_Obstacle:
                self.obstacles.append(self.temp_rect)
                self.emit_new_obstacles()
            else:
                self.scene.removeItem(self.temp_rect)
            self.temp_rect = None
            self.start_pos_of_new_rectangle = None
        self.drag_start_pos = None

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        if self.selected_item and event.key() in (Qt.Key_Backspace, Qt.Key_Delete):
            if isinstance(self.selected_item, WaypointCompositeItem):
                self.remove_waypoint(self.selected_item)
            elif isinstance(self.selected_item, DraggableRect):
                self.obstacles.remove(self.selected_item)
                self.emit_new_obstacles()
            if self.selected_item in self.scene.items():
                self.scene.removeItem(self.selected_item)
            self.selected_item = None

    def wheelEvent(self, event):
        if self.mode == Mode.Zoom:
            super(PathCanvas, self).wheelEvent(event)
        elif self.selected_item and isinstance(
            self.selected_item, WaypointCompositeItem
        ):
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