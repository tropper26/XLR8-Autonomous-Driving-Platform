from control.base_params import BaseControllerParams
from dto.form_dto import FormDTO


class PurePursuitParams(BaseControllerParams, FormDTO):
    def __init__(self, forward_gain, base_look_ahead_distance, kp, tuning_params=None):
        super().__init__()
        self.forward_gain = forward_gain
        self.base_look_ahead_distance = base_look_ahead_distance
        self.kp = kp
        self.tuning_params = tuning_params

    def attributes_to_ignore(self):
        return ["tuning_params"]

    @classmethod
    def from_dict(cls, saved_dict: dict):
        return cls(
            forward_gain=saved_dict["forward_gain"],
            base_look_ahead_distance=saved_dict["base_look_ahead_distance"],
            kp=saved_dict["kp"],
        )

    def __str__(self):
        return f"forward_gain: {self.forward_gain}, base_look_ahead_distance: {self.base_look_ahead_distance}, Kp: {self.kp}"

    __repr__ = __str__