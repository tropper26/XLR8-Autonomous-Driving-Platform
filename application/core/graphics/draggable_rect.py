from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtGui import QColor, QBrush, QCursor
from PyQt5.QtWidgets import (
    QGraphicsRectItem,
    QGraphicsItem,
    QApplication,
)


class SignalEmitter(QObject):
    valueChanged = pyqtSignal()


class DraggableRect(QGraphicsRectItem):
    def __init__(self, x, y, width, height, color=QColor("black")):
        super().__init__(x, y, width, height)
        self.signals = SignalEmitter()
        self.setBrush(QBrush(color))
        self.setFlags(
            QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

    @property
    def valueChanged(self):
        return self.signals.valueChanged

    def get_scene_rect(self):
        return self.mapRectToScene(self.rect())
        # return self.rect()

    def hoverEnterEvent(self, event):
        QApplication.setOverrideCursor(
            QCursor(Qt.PointingHandCursor)
        )  # Change cursor on hover enter
        self.setBrush(QBrush(QColor("gray")))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        QApplication.restoreOverrideCursor()
        if not self.isSelected():
            self.setBrush(QBrush(QColor("black")))
        super().hoverLeaveEvent(event)

    def setSelected(self, selected):
        if selected:
            self.setBrush(QBrush(QColor("red")))
        else:
            self.setBrush(QBrush(QColor("black")))

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            self.valueChanged.emit()
        elif change == QGraphicsItem.ItemSelectedChange:
            self.setSelected(value)
        return super(DraggableRect, self).itemChange(change, value)