from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsRectItem, QApplication
from PyQt5.QtGui import QBrush, QColor
import sys

class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super(ZoomableGraphicsView, self).__init__(parent)

        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self.zoomFactor = 1.15  # Define how much to zoom on each scroll
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)  # Zoom relative to the mouse position

    def wheelEvent(self, event):
        # Zoom in
        if event.angleDelta().y() > 0:
            self.scale(self.zoomFactor, self.zoomFactor)
        # Zoom out
        else:
            self.scale(1 / self.zoomFactor, 1 / self.zoomFactor)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Create a view and set the scene
    view = ZoomableGraphicsView()
    view.setWindowTitle("Zoomable QGraphicsView with Shapes")
    view.resize(800, 600)
    view.show()

    # Add some rectangles to the scene
    for i in range(5):
        rect = QGraphicsRectItem(0, 0, 100, 100)  # A square
        rect.setBrush(QBrush(QColor(100 + i * 30, 0, 200 - i * 40)))  # Set different colors
        rect.setPos(i * 120, i * 120)  # Spread out the squares
        view.scene.addItem(rect)

    sys.exit(app.exec_())