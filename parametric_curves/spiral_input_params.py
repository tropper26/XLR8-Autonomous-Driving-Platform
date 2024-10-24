class SpiralInputParams:
    __slots__ = ('x_0', 'y_0', 'psi_0', 'k_0', 'x_f', 'y_f', 'psi_f', 'k_f', 'k_max')

    def __init__(self, x_0, y_0, psi_0, k_0, x_f, y_f, psi_f, k_f, k_max):
        self.x_0 = x_0
        self.y_0 = y_0
        self.psi_0 = psi_0
        self.k_0 = k_0
        self.x_f = x_f
        self.y_f = y_f
        self.psi_f = psi_f
        self.k_f = k_f
        self.k_max = k_max

    def as_tuple(self):
        return self.x_0, self.y_0, self.psi_0, self.k_0, self.x_f, self.y_f, self.psi_f, self.k_f, self.k_max

    def __repr__(self):
        return (f"SpiralInputParams(x_0={self.x_0}, y_0={self.y_0}, psi_0={self.psi_0}, "
                f"x_f={self.x_f}, y_f={self.y_f}, psi_f={self.psi_f}, k_0={self.k_0}, "
                f"k_f={self.k_f}, k_max={self.k_max})")