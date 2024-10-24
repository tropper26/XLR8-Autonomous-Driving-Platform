import json
from collections import namedtuple

import numpy as np

from dto.vector3 import Vector3

GPS_INFO = namedtuple("GPS_INFO", ["x", "y", "z", "confidence"])


def convert_quaternion_to_euler_angles(quaternion):
    w, x, y, z = quaternion
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    # pitch = np.arcsin(2 * (w * y - z * x))
    pitch = -np.pi / 2 + 2 * np.arctan2(
        np.sqrt(1 + 2 * (w * y - z * x)), np.sqrt(1 - 2 * (w * y - z * x))
    )
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    return Vector3((roll, pitch, yaw))


class SimulationSensorInfo:
    def __init__(self, filename: str):
        self.filename = filename
        self.last_file_position = 0  # Initial file position

        self.imu_acceleration: Vector3 = Vector3((None, None, None))
        self.imu_euler_angles: Vector3 = Vector3((None, None, None))
        self.gps_position: GPS_INFO = GPS_INFO(None, None, None, None)
        self.steering_angle = 0
        self._gps_changed = False
        self._imu_changed = False

    def update(self, current_time: float):
        with open(self.filename, "r") as file:
            file.seek(self.last_file_position)

            while True:
                line = file.readline()

                if not line:
                    # End of file
                    break

                data = json.loads(line)
                if (
                    "timestamp" in data and data["timestamp"] <= current_time
                ) or (  # For the old logs
                    "timestamp_ms" in data
                    and data["timestamp_ms"] - 1709134671247 <= current_time
                ):
                    self.last_file_position = file.tell()
                    if "Imu" in data:
                        if "acceleration" in data["Imu"]:
                            self.imu_acceleration = Vector3(data["Imu"]["acceleration"])
                            self._imu_changed = True

                        if "quaternion" in data["Imu"]:
                            if None not in data["Imu"]["quaternion"]:
                                if (
                                    data["Imu"]["quaternion"][0] ** 2
                                    + data["Imu"]["quaternion"][1] ** 2
                                    + data["Imu"]["quaternion"][2] ** 2
                                    + data["Imu"]["quaternion"][3] ** 2
                                    - 1
                                    > 1e-3
                                ):
                                    print(
                                        "Quaternion is not normalized: ",
                                        data["Imu"]["quaternion"],
                                        data["Imu"]["quaternion"][0] ** 2
                                        + data["Imu"]["quaternion"][1] ** 2
                                        + data["Imu"]["quaternion"][2] ** 2
                                        + data["Imu"]["quaternion"][3] ** 2,
                                    )
                                else:
                                    quaternion = data["Imu"]["quaternion"]
                                    self.imu_euler_angles = (
                                        convert_quaternion_to_euler_angles(quaternion)
                                    )
                                    self.imu_euler_angles = Vector3(
                                        (
                                            self.imu_euler_angles.x,
                                            self.imu_euler_angles.y,
                                            self.imu_euler_angles.z + np.pi / 2,
                                        )  # The imu is rotated 90 degrees
                                    )
                                    self._imu_changed = True
                    elif "Gps" in data:
                        self.gps_position = GPS_INFO(
                            data["Gps"]["x"],
                            data["Gps"]["y"],
                            data["Gps"]["z"],
                            data["Gps"]["confidence"],
                        )
                        self._gps_changed = True
                    elif "SteeringAngle" in data:
                        self.steering_angle = data["SteeringAngle"]
                        self._imu_changed = True
                    elif "Velocity" in data:
                        pass
                    elif "Distance" in data:
                        pass
                    else:
                        print("Unknown sensor type: ", data)
                else:
                    break

    def gps_changed_since_last_check(self):
        if self._gps_changed:
            self._gps_changed = False
            return True
        return False

    def imu_changed_since_last_check(self):
        if self._imu_changed:
            self._imu_changed = False
            return True
        return False

    @property
    def roll(self):
        return self.imu_euler_angles.x

    @property
    def pitch(self):
        return self.imu_euler_angles.y

    @property
    def yaw(self):
        return self.imu_euler_angles.z

    __repr__ = (
        __str__
    ) = (
        lambda self: f"Imu: {self.imu_acceleration}, {self.imu_euler_angles}, Gps: {self.gps_position}"
    )