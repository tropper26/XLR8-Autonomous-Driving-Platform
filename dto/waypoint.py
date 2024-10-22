import numpy as np


class Waypoint:
    def __init__(self, x: float, y: float, id: int = None):
        self.id = id
        self._data = np.array([x, y])

    @property
    def x(self):
        return self._data[0]

    @property
    def y(self):
        return self._data[1]

    def __getitem__(self, index):
        return self._data[index]

    def __str__(self) -> str:
        return f"Waypoint(id={self.id}, x={self.x}, y={self.y})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other) -> bool:
        if not isinstance(other, Waypoint):
            return NotImplemented
        return np.array_equal(self._data, other._data)


class WaypointWithHeading(Waypoint):
    def __init__(self, x: float, y: float, heading: float, id: int = None):
        super().__init__(x, y, id)
        self._data = np.array([x, y, heading])

    @property
    def heading(self):
        return self._data[2]

    def __str__(self) -> str:
        return f"WaypointWithHeading(id={self.id}, x={self.x}, y={self.y}, heading={self.heading})"

    def __hash__(self) -> int:
        return hash(tuple(self._data))