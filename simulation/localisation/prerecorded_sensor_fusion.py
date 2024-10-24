import numpy as np

from simulation.localisation.kalman_filter import KalmanFilter
from simulation.localisation.sensor_info import SimulationSensorInfo
from state_space.inputs.control_action import ControlAction
from state_space.models.generic_model import GenericModel

from state_space.states.state import State
from vehicle.vehicle_params import VehicleParams

np.set_printoptions(
    precision=3,
    linewidth=9999999999,
)


class PrerecordedSensorFusion:
    def __init__(
        self,
        sensor_log_file_path,
        model: GenericModel[State],
        vehicle_info: VehicleParams,
        sampling_time: float,
    ):
        self.vehicle_info = vehicle_info
        self.sampling_time = sampling_time
        self.kalman_filter = KalmanFilter(model=model)
        self.current_sensor_info = SimulationSensorInfo(sensor_log_file_path)

    def initialize(
        self,
        initial_state: State,
        starting_control_input: ControlAction,
        sampling_time: float,
    ):
        self.kalman_filter.initialize(
            initial_state=initial_state,
            starting_control_input=starting_control_input,
            sampling_time=sampling_time,
        )

    def run_iteration(self, current_time):
        self.current_sensor_info.update(current_time)

        if self.current_sensor_info.imu_changed_since_last_check():
            self.kalman_filter.prediction_step(
                observed_longitudinal_a=self.current_sensor_info.imu_acceleration.x,
                observed_heading_angle=self.current_sensor_info.imu_euler_angles.z,
                observed_steering_angle=self.current_sensor_info.steering_angle,
            )
        if self.current_sensor_info.gps_changed_since_last_check():
            self.kalman_filter.correction_step(
                observed_position_x=self.current_sensor_info.gps_position.x,
                observed_position_y=self.current_sensor_info.gps_position.y,
            )

        return self.kalman_filter.state_space.X

    def run_iteration_sensors_only(self, current_time):
        self.current_sensor_info.update(current_time)

        sensor_state = State(
            X=self.current_sensor_info.gps_position.x
            if self.current_sensor_info.gps_position.x
            else 0,
            Y=self.current_sensor_info.gps_position.y
            if self.current_sensor_info.gps_position.y
            else 0,
            Psi=self.current_sensor_info.imu_euler_angles.z
            if self.current_sensor_info.imu_euler_angles.z
            else 0,
            x_dot=0,
            y_dot=0,
            psi_dot=0,
        )

        return sensor_state