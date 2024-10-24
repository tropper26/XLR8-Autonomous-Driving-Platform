import numpy as np


class Rectangle:
    _id_counter = 0

    def __init__(self, x, y, width, height, id=None, lifetime_seconds=None):
        self.x = x  # x-coordinate of the bottom-left corner
        self.y = y  # y-coordinate of the bottom-left corner
        self.width = width
        self.height = height
        if id is not None:
            self.id = id
        else:
            self.id = Rectangle._id_counter
            Rectangle._id_counter += 1

        if lifetime_seconds is not None:
            self.lifetime_seconds = lifetime_seconds
        else:
            self.lifetime_seconds = float(np.random.randint(2, 4))

    def decrement_lifetime(self, dt_seconds: float):
        self.lifetime_seconds -= dt_seconds
        print(f"Decrementing lifetime of rectangle {self.id} to {self.lifetime_seconds}:{dt_seconds}")

    @property
    def x_center(self):
        return self.x + self.width / 2

    @property
    def y_center(self):
        return self.y + self.height / 2

    def intersects_circles(
            self, circle_centers: np.ndarray, circle_radii: np.ndarray | float
    ) -> np.ndarray:
        """
        Check if any circle in the array intersects this rectangle using vectorized operations.

        Returns:
        np.ndarray: A boolean array where each element is True if the corresponding circle intersects the rectangle.
        """
        # Calculate the closest points on the rectangle to each circle center
        closest_x = np.clip(circle_centers[:, 0], self.x, self.x + self.width)
        closest_y = np.clip(circle_centers[:, 1], self.y, self.y + self.height)

        # Calculate distances from circle centers to these closest points
        distances_squared = (circle_centers[:, 0] - closest_x) ** 2 + (
                circle_centers[:, 1] - closest_y
        ) ** 2

        # Check for intersections
        return distances_squared <= circle_radii ** 2

    def __str__(self):
        return f"Rectangle(x={self.x}, y={self.y}, width={self.width}, height={self.height})"

    def __repr__(self):
        return f"Rectangle(x={self.x}, y={self.y}, width={self.width}, height={self.height})"