from collections import namedtuple

OutputWeights = namedtuple(
    "OutputWeights", ["velocity", "heading_angle", "X_global", "Y_global"]
)

InputWeights = namedtuple("InputWeights", ["acceleration", "steering_angle"])


CostFunctionWeightMatrices = namedtuple("CostFunctionWeightMatrices", ["Q", "S", "R"])


class Weights:
    def __init__(
        self,
        Q_weights: OutputWeights,
        S_weights: OutputWeights,
        R_weights: InputWeights,
    ):
        self.Q_weights = Q_weights
        self.S_weights = S_weights
        self.R_weights = R_weights


def define_weights():
    Q_weights = OutputWeights(
        velocity=1.0, heading_angle=200.0, X_global=50.0, Y_global=50.0
    )
    S_weights = Q_weights
    R_weights = InputWeights(acceleration=1.0, steering_angle=100.0)
    w1 = Weights(Q_weights, S_weights, R_weights)
    w2 = w1
    Q_weights = OutputWeights(
        velocity=100.0, heading_angle=20000.0, X_global=1000.0, Y_global=1000.0
    )
    S_weights = Q_weights
    R_weights = InputWeights(acceleration=100.0, steering_angle=1)
    w3 = Weights(Q_weights, S_weights, R_weights)
    w4 = w3
    Q_weights = OutputWeights(
        velocity=1.0, heading_angle=200.0, X_global=10000.0, Y_global=10000.0
    )
    S_weights = OutputWeights(
        velocity=5.0, heading_angle=300.0, X_global=50000.0, Y_global=50000.0
    )
    R_weights = InputWeights(acceleration=1.0, steering_angle=100.0)
    w5 = Weights(Q_weights, S_weights, R_weights)

    return w1, w2, w3, w4, w5