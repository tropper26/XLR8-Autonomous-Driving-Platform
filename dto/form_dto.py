from abc import ABC, abstractmethod


class FormDTO(ABC):
    @abstractmethod
    def attributes_to_ignore(self) -> list[str]:
        pass