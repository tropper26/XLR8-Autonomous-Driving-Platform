from dto.form_dto import FormDTO


class AutoTunerParams(FormDTO):
    def __init__(
        self,
        learning_rate=0.0005,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.99,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def attributes_to_ignore(self):
        return []