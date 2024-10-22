from abc import ABC, abstractmethod

from control.reinforcement_learning.tuning.auto_tuner_params import AutoTunerParams


class BaseControllerParams(ABC):
    def __init__(self, tuning_params: AutoTunerParams = None):
        self.tuning_params = tuning_params

    @classmethod
    @abstractmethod
    def from_dict(cls, saved_dict: dict):
        return cls()