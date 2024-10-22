import numpy as np


def compute_path_frame_error(
    X_path,
    Y_path,
    Psi_path,
    X_vehicle,
    Y_vehicle,
    Psi_vehicle,
    x_dot_path=None,
    x_dot_vehicle=None,
):
    (
        x_vehicle_path_frame,
        y_vehicle_path_frame,
        x_dot_vehicle_path_frame,
    ) = inertial_to_path_frame(
        x_path=X_path,
        y_path=Y_path,
        psi_path=Psi_path,
        x_vehicle=X_vehicle,
        y_vehicle=Y_vehicle,
        psi_vehicle=Psi_vehicle,
        x_dot_vehicle=x_dot_vehicle,
    )
    error_X_path_frame = -x_vehicle_path_frame
    error_Y_path_frame = (
        -y_vehicle_path_frame
    )  # would be  Y_p - front_Y_v but Y_p is 0 in path frame
    error_psi = (
        Psi_path - Psi_vehicle
    )  # would be Psi_v - Psi_p but Psi_p is 0 in path frame
    if x_dot_path is None:
        return error_X_path_frame, error_Y_path_frame, error_psi, None

    error_x_dot_path_frame = x_dot_path - x_dot_vehicle_path_frame

    return error_X_path_frame, error_Y_path_frame, error_psi, error_x_dot_path_frame


def has_mixed_types(arr):
    first_type = type(arr[0])

    for element in arr:
        if type(element) is not first_type:
            print("first element: ", arr[0], "type: ", type(arr[0]))
            print("element: ", element, "type: ", type(element))
            return True

    return False


def inertial_to_path_frame(
    x_path, y_path, psi_path, x_vehicle, y_vehicle, psi_vehicle, x_dot_vehicle=None
) -> (float, float, float, float | None):
    if isinstance(psi_path, (float, int)):
        cos_inertial_to_path = np.cos(-psi_path)
        sin_inertial_to_path = np.sin(-psi_path)
    else:
        cos_inertial_to_path = np.cos(-psi_path.astype(np.float64))
        sin_inertial_to_path = np.sin(-psi_path.astype(np.float64))

    # Translate and rotate coordinates
    x_vehicle_path_frame = (x_vehicle - x_path) * cos_inertial_to_path - (
        y_vehicle - y_path
    ) * sin_inertial_to_path
    y_vehicle_path_frame = (x_vehicle - x_path) * sin_inertial_to_path + (
        y_vehicle - y_path
    ) * cos_inertial_to_path

    if x_dot_vehicle is None:
        return x_vehicle_path_frame, y_vehicle_path_frame, None

    vx = x_dot_vehicle * np.cos(psi_vehicle)
    vy = x_dot_vehicle * np.sin(psi_vehicle)
    x_dot_vehicle_path_frame = vx * cos_inertial_to_path - vy * sin_inertial_to_path
    y_dot_vehicle_path_frame = vx * sin_inertial_to_path + vy * cos_inertial_to_path

    return (
        x_vehicle_path_frame,
        y_vehicle_path_frame,
        x_dot_vehicle_path_frame,
    )


def path_to_inertial_frame(
    x_path,
    y_path,
    psi_path,
    x_vehicle_path_frame,
    y_vehicle_path_frame,
    x_dot_vehicle_path_frame=None,
):
    cos_path_to_inertial = np.cos(psi_path)
    sin_path_to_inertial = np.sin(psi_path)

    x_vehicle = (
        x_vehicle_path_frame * cos_path_to_inertial
        - y_vehicle_path_frame * sin_path_to_inertial
        + x_path
    )
    y_vehicle = (
        x_vehicle_path_frame * sin_path_to_inertial
        + y_vehicle_path_frame * cos_path_to_inertial
        + y_path
    )

    if x_dot_vehicle_path_frame is None:
        return x_vehicle, y_vehicle, None

    x_dot_vehicle = x_dot_vehicle_path_frame * cos_path_to_inertial
    y_dot_vehicle = x_dot_vehicle_path_frame * sin_path_to_inertial

    return x_vehicle, y_vehicle, x_dot_vehicle