from numpy import ndarray


class StateSpaceMatrices:
    """
    The state space matrices of the model.
    """

    def __init__(self, A: ndarray, B: ndarray, C: ndarray, D: ndarray):
        self.A = A
        self.B = B
        self.C = C
        self.D = D