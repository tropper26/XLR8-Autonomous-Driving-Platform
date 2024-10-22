from control.base_params import BaseControllerParams
from control.mpc.weights import define_weights
from control.reinforcement_learning.tuning.auto_tuner_params import AutoTunerParams
from dto.form_dto import FormDTO


class MPCParams(BaseControllerParams, FormDTO):
    """
    Constants related to Model Predictive Control (MPC).
    """

    def __init__(
        self,
        nr_outputs: int,
        nr_inputs: int,
        horizon_period: int,
        sampling_time: float,
        tuning_params: AutoTunerParams = None,
    ):
        super().__init__()
        self.nr_outputs = nr_outputs
        self.nr_inputs = nr_inputs
        self.horizon_period = horizon_period
        self.sampling_time = sampling_time
        self.tuning_params = tuning_params
        w1, w2, w3, w4, w5 = define_weights()
        self.weights = w3

    def attributes_to_ignore(self):
        return [
            "tuning_params",
            "weights",
        ]  # TODO weights should be handled in a different way

    @classmethod
    def from_dict(cls, saved_dict: dict):
        return cls(
            nr_outputs=saved_dict["nr_outputs"],
            nr_inputs=saved_dict["nr_inputs"],
            horizon_period=saved_dict["horizon_period"],
            sampling_time=saved_dict["sampling_time"],
        )

    def __str__(self):
        return f"nr_outputs: {self.nr_outputs}, nr_inputs: {self.nr_inputs}, horizon_period: {self.horizon_period}"

    __repr__ = __str__