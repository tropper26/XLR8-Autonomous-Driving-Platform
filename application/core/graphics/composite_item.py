import numpy as np
from PyQt5.QtCore import QPointF, Qt, QRectF, QObject, pyqtSignal
from PyQt5.QtGui import QPen, QColor, QPainter, QCursor
from PyQt5.QtWidgets import (
    QGraphicsItem,
    QGraphicsTextItem,
    QGraphicsLineItem,
    QGraphicsEllipseItem,
    QStyleOptionGraphicsItem,
    QApplication,
)


class SignalEmitter(QObject):
    valueChanged = pyqtSignal()


class CompositeItem(QGraphicsItem):
    def __init__(self, length=100, initial_rotation_radians=0):
        super(CompositeItem, self).__init__()
        self.signals = SignalEmitter()
        self.length = length

        self.selected = False

        self.angle_radians = initial_rotation_radians
        normalized_angle_degrees = (
            -np.mod(np.degrees(self.angle_radians) + 180, 360) + 180
        )  # Normalize angle to be between 180 and -180 degrees
        self.setRotation(normalized_angle_degrees)

        self.angle_label = QGraphicsTextItem("", self)
        self.angle_label.setDefaultTextColor(Qt.red)
        self.updateAngleLabel(-int(normalized_angle_degrees))

        self.default_pen = QPen(Qt.black, 3)
        self.hover_pen = QPen(QColor(0, 127, 255), 3)
        self.selected_pen = QPen(QColor(255, 0, 0), 3)

        self.segment = QGraphicsLineItem(-self.length / 2, 0, self.length / 2, 0, self)
        self.segment.setPen(self.default_pen)

        self.point = QGraphicsEllipseItem(-5, -5, 10, 10, self)
        self.point.setPen(self.default_pen)

        self.setFlags(
            QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemSendsGeometryChanges
        )

    @property
    def valueChanged(self):
        return self.signals.valueChanged

    def setPos(self, point: QPointF):
        super(CompositeItem, self).setPos(point)

    def boundingRect(self):
        return QRectF(-self.length / 2 - 10, -15, self.length + 20, 30)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget=None):
        # if option.state & QStyle.State_Selected:
        painter.setPen(QPen(QColor(255, 200, 0), 1, Qt.DashLine))
        painter.drawRect(self.boundingRect())

    def updateAngleLabel(self, degrees):
        self.angle_label.setPlainText(f"{degrees}°")
        self.angle_label.setPos(self.length / 2 + 5, -15)
        self.angle_label.setRotation(-self.rotation())  # Rotate text to be upright

    def rotate(self, delta_angle_degrees):
        angle_degrees = self.rotation() + delta_angle_degrees
        self.setRotation(angle_degrees)

        normalized_angle_degrees = (
            -np.mod(angle_degrees + 180, 360) + 180
        )  # Normalize angle to be between 180 and -180 degrees
        self.updateAngleLabel(int(normalized_angle_degrees))

        self.angle_radians = np.radians(normalized_angle_degrees)
        self.valueChanged.emit()

    def setSelected(self, is_selected):
        self.selected = is_selected
        pen = self.selected_pen if is_selected else self.default_pen
        self.segment.setPen(pen)
        self.point.setPen(pen)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            self.valueChanged.emit()
        elif change == QGraphicsItem.ItemSelectedChange:
            self.setSelected(value)
        return super(CompositeItem, self).itemChange(change, value)

    def hoverEnterEvent(self, event):
        QApplication.setOverrideCursor(
            QCursor(Qt.PointingHandCursor)
        )  # Change cursor on hover enter
        self.segment.setPen(self.hover_pen)
        self.point.setPen(self.hover_pen)
        self.update()

    def hoverLeaveEvent(self, event):
        QApplication.restoreOverrideCursor()
        if not self.isSelected():
            self.segment.setPen(self.default_pen)
            self.point.setPen(self.default_pen)
            self.update()