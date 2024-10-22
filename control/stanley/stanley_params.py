from control.base_params import BaseControllerParams
from dto.form_dto import FormDTO


class StanleyParams(BaseControllerParams, FormDTO):
    def __init__(self, k: float, kp: float):
        """
        :param k: Proportional gain for cross-track error in the Stanley controller.
        :param kp: Proportional gain for velocity error.
        """
        super().__init__()
        self.k = k  # Gain for cross-track error correction
        self.kp = kp  # Gain for proportional speed control

    def attributes_to_ignore(self):
        return ["tuning_params"]

    @classmethod
    def from_dict(cls, saved_dict: dict):
        return cls(
            k=saved_dict["k"],
            kp=saved_dict["kp"],
        )

    def __str__(self):
        return f"Kp: {self.kp}" f"k: {self.k}"

    __repr__ = __str__