
class LaneWidthInfos:
    def __init__(self, left_lane_count: int, right_lane_count: int, lane_widths: list[float]):
        self.left_lane_count = left_lane_count
        self.right_lane_count = right_lane_count
        self.lane_widths = lane_widths
