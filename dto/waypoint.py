class Waypoint:
    __slots__ = ('x', 'y', 'index_in_route')

    def __init__(self, x: float, y: float, index_in_route: int = None):
        self.index_in_route = index_in_route
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return f"Waypoint(id={self.index_in_route}, x={self.x}, y={self.y})"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other) -> bool:
        if not isinstance(other, Waypoint):
            return NotImplemented
        return self.x == other.x and self.y == other.y

class WaypointWithHeading(Waypoint):
    __slots__ = ('heading')

    def __init__(self, x: float, y: float, heading: float, index_in_route: int = None):
        super().__init__(x, y, index_in_route)
        self.heading = heading

    def __str__(self) -> str:
        return f"WaypointWithHeading(id={self.index_in_route}, x={self.x}, y={self.y}, heading={self.heading})"