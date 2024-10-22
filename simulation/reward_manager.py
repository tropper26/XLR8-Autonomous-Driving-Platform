import numpy as np


class RewardManager:
    def __init__(
        self,
        destination_reached_reward: float,
        out_of_bounds_penalty: float,
        max_lateral_error_threshold: float,
        max_lateral_error_penalty: float,
        max_heading_error_threshold: float,
        max_heading_error_penalty: float,
        max_velocity_error_threshold: float,
        max_velocity_error_penalty: float,
        amplitude_Y: float = 1,
        variance_Y: float = 0.05,
        amplitude_Psi: float = 1,
        variance_Psi: float = np.sqrt(0.005),
        amplitude_x_dot: float = 1,
        variance_x_dot: float = np.sqrt(0.1),
    ):
        self._destination_reached_reward = destination_reached_reward
        self._out_of_bounds_penalty = out_of_bounds_penalty
        self.max_lateral_error_threshold = max_lateral_error_threshold
        self._max_lateral_error_penalty = max_lateral_error_penalty
        self.max_heading_error_threshold = max_heading_error_threshold
        self._max_heading_error_penalty = max_heading_error_penalty
        self.max_velocity_error_threshold = max_velocity_error_threshold
        self._max_velocity_error_penalty = max_velocity_error_penalty
        self.amplitude_Y = amplitude_Y
        self.variance_Y = variance_Y
        self.amplitude_Psi = amplitude_Psi
        self.variance_Psi = variance_Psi
        self.amplitude_x_dot = amplitude_x_dot
        self.variance_x_dot = variance_x_dot

    def lateral_error_penalty(self, error_Y_path_frame) -> (float, str):
        return (
            self._max_lateral_error_penalty,
            f"{self._max_lateral_error_penalty}, Lateral error {abs(error_Y_path_frame)} > {self.max_lateral_error_threshold}.",
        )

    def heading_error_penalty(self, error_psi) -> (float, str):
        return (
            self._max_heading_error_penalty,
            f"{self._max_heading_error_penalty}, Heading error {abs(np.degrees(error_psi))} > {np.degrees(self.max_heading_error_threshold)} (degrees).",
        )

    def velocity_error_penalty(self, error_x_dot_path_frame) -> (float, str):
        return (
            self._max_velocity_error_penalty,
            f"{self._max_velocity_error_penalty}, x_dot error {abs(error_x_dot_path_frame)} > {self.max_velocity_error_threshold}.",
        )

    def compute_reward(
        self, error_Y_path_frame, error_psi, error_x_dot_path_frame, delta_d
    ) -> (float, str):
        g_Y = self.compute_gaussian_like_value(
            self.amplitude_Y, self.variance_Y, error_Y_path_frame
        )
        g_Psi = self.compute_gaussian_like_value(
            self.amplitude_Psi, self.variance_Psi, error_psi
        )
        g_x_dot = self.compute_gaussian_like_value(
            self.amplitude_x_dot, self.variance_x_dot, error_x_dot_path_frame
        )

        reward = g_Y * (1 + (g_Psi + g_x_dot) * (1 + 1 / (1 + delta_d)))
        return (
            reward,
            f"{reward:.2f} = {g_Y:.2f} * (1 + ({g_Psi:.2f} + {g_x_dot:.2f}) * (1 + 1 / (1 + {delta_d:.2f})).",
        )

    def compute_reward_without_velocity(
        self, error_Y_path_frame, error_psi, delta_d
    ) -> (float, str):
        g_Y = self.compute_gaussian_like_value(
            self.amplitude_Y, self.variance_Y, error_Y_path_frame
        )
        g_Psi = self.compute_gaussian_like_value(
            self.amplitude_Psi, self.variance_Psi, error_psi
        )

        reward = g_Y * (1 + g_Psi * (1 + 1 / (1 + delta_d)))
        return reward

    def out_of_bounds_penalty(self) -> (float, str):
        return (
            self._out_of_bounds_penalty,
            f"{self._out_of_bounds_penalty}, Out of bounds.",
        )

    def destination_reached_reward(self) -> (float, str):
        return (
            self._destination_reached_reward,
            f"{self._destination_reached_reward}, Destination reached.",
        )

    def compute_gaussian_like_value(self, a: float, b: float, value) -> float:
        """
        Compute the Gaussian-like function value.

        Parameters:
        - a (float): The amplitude of the Gaussian curve.
        - b (float): The variance (spread) of the Gaussian curve. Must be positive.
        - value: The input value for which the Gaussian function is calculated.

        Returns:
        - float: The result of the Gaussian function calculation.

        """
        if b <= 0:
            raise ValueError(
                "Parameter 'b' must be positive as it represents the variance."
            )
        if isinstance(value, (float, int)):
            return a * np.exp(-(value**2) / (2 * b))
        else:
            return a * np.exp(-(value.astype(np.float64) ** 2) / (2 * b))