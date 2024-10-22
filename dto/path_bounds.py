class WorldBounds:
    def __init__(self, min_X: float, max_X: float, min_Y: float, max_Y: float):
        self.min_X = min_X
        self.max_X = max_X
        self.min_Y = min_Y
        self.max_Y = max_Y

    def check_in_bounds(self, X: float, Y: float) -> bool:
        return self.min_X <= X <= self.max_X and self.min_Y <= Y <= self.max_Y