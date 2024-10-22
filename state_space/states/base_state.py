from abc import ABC, abstractmethod

import pandas as pd
from numpy import array


class BaseState(ABC):
    @abstractmethod
    def __init__(self, column_vector_form=None):
        pass

    @abstractmethod
    def __add__(self, other):
        pass

    @abstractmethod
    def add_column_vector(self, other: array):
        pass

    @abstractmethod
    def as_series(self) -> pd.Series:
        pass