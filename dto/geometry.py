import numpy as np


class Rectangle:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

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
        return distances_squared <= circle_radii**2

    def __str__(self):
        return f"Rectangle(x={self.x}, y={self.y}, width={self.width}, height={self.height})"

    def __repr__(self):
        return f"Rectangle(x={self.x}, y={self.y}, width={self.width}, height={self.height})"