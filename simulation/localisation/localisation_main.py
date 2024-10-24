import cv2
import numpy as np
from tqdm import tqdm

from simulation.localisation.prerecorded_sensor_fusion import PrerecordedSensorFusion
from state_space.inputs.control_action import ControlAction
from state_space.models.kinematic_bicycle_model import KinematicBicycleModel
from state_space.states.state import State
from vehicle.vehicle_params import VehicleParams


def create_video_writer(cap, output_path):
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    return cv2.VideoWriter(
        output_path,
        fourcc,
        cap.get(cv2.CAP_PROP_FPS),
        (int(cap.get(3)), int(cap.get(4))),
    )


def find_yellow_pixels(hsv_frame):
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
    return np.column_stack(np.where(yellow_mask > 0))


def kmeans_clustering(yellow_pixels, num_clusters=2):
    _, labels, centers = cv2.kmeans(
        yellow_pixels.astype(np.float32),
        num_clusters,
        None,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.2),
        attempts=10,
        flags=cv2.KMEANS_RANDOM_CENTERS,
    )
    centers = np.uint16(centers)
    return centers


def put_multiple_text(frame, texts, origins, colors, font_scale=1, thickness=2):
    for text, origin, color in zip(texts, origins, colors):
        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
            False,
        )


def draw_cluster_centers(frame, centers):
    for center in centers:
        cv2.circle(frame, (center[1], center[0]), 5, (0, 0, 255), -1)


def calculate_overall_mean_coords(centers):
    return np.mean(centers, axis=0)


def process_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow_pixels = find_yellow_pixels(hsv)

    height, width, channels = frame.shape
    y_meters_max, x_meters_max = 285, 532.5

    draw_borders(frame, height, width, x_meters_max, y_meters_max)

    camera_pixels_x, camera_pixels_y = 0, 0

    if len(yellow_pixels) > 0:
        centers = kmeans_clustering(
            yellow_pixels
        )  # for center first y (height), second x (width)
        try:
            draw_cluster_centers(frame, centers)
            (camera_pixels_y, camera_pixels_x) = calculate_overall_mean_coords(centers)
        except Exception as e:
            print("Draw centers failed: ", e)

    return frame, camera_pixels_x, camera_pixels_y


def draw_borders(frame, height, width, x_meters_max, y_meters_max):
    offset = 50
    cv2.putText(
        frame,
        f"{0, y_meters_max}",
        (offset, height - offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        1 / 2,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
        False,
    )
    cv2.putText(
        frame,
        f"{x_meters_max, 0}",
        (width - 2 * offset, offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        1 / 2,
        (255, 0, 0),
        1,
        cv2.LINE_AA,
        False,
    )
    cv2.putText(
        frame,
        f"{x_meters_max, y_meters_max}",
        (width - 2 * offset, height - offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        1 / 2,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
        False,
    )


def draw_error(frame, avg_error, rolling_error, current_error):
    texts = [
        f"Avg     Er: {round(avg_error, 2)} cm",
        f"Rolling  Er: {round(rolling_error, 2)} cm",
        f"Current Er: {round(current_error, 2)} cm",
    ]
    orgs = [(1550, 100), (1550, 150), (1550, 200)]
    put_multiple_text(frame, texts, orgs, [(0, 0, 255)] * 3, thickness=4)


def put_state_on_frame(
    frame,
    state_meters: State,
    starting_height,
    starting_width,
    height_offset,
    color=(100, 255, 255),
):
    texts = [
        f"X: {round(state_meters.X, 2)} m",
        f"Y: {round(state_meters.Y, 2)} m",
        f"psi: {round(state_meters.Psi, 2)} rad, {round(np.rad2deg(state_meters.Psi), 2)} deg",
        f"x_dot: {round(state_meters.x_dot, 2)} m/s",
        f"y_dot: {round(state_meters.y_dot, 2)} m/s",
        f"psi_dot: {round(state_meters.psi_dot, 2)} rad/s",
    ]
    origins = [
        (starting_width, starting_height + i * height_offset) for i in range(len(texts))
    ]
    put_multiple_text(frame, texts, origins, [color] * len(texts))


def draw_circles_for_positions(
    frame, positions, color, text_color, width, height, position_in_meters=True
):
    if len(positions) == 0:
        raise ValueError("Positions list is empty.")

    pixels_x, pixels_y = None, None
    meters_x, meters_y = None, None

    for pos in positions:
        if position_in_meters:
            pixels_x, pixels_y = meters_to_pixels(pos[0], pos[1], width, height)
            pixels_x, pixels_y = int(pixels_x), int(pixels_y)
            meters_x, meters_y = round(pos[0], 3), round(pos[1], 3)
        else:
            pixels_x, pixels_y = int(pos[0]), int(pos[1])
            meters_x, meters_y = pixels_to_meters(pixels_x, pixels_y, width, height)
            meters_x, meters_y = round(meters_x, 3), round(meters_y, 3)

        cv2.circle(
            frame,
            (pixels_x, pixels_y),
            10,
            color,
            -1,
        )

    cv2.putText(
        frame,
        f"{meters_x}, {meters_y}",
        (pixels_x + 50, pixels_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        text_color,
        2,
        cv2.LINE_AA,
        False,
    )

    return pixels_x, pixels_y, meters_x, meters_y


def pixels_to_cm(x_pixels, y_pixels, frame_width, frame_height):
    koefficient_x = 532.5 / frame_width
    x_cm = round(koefficient_x * x_pixels, 3)
    koefficient_y = 285 / frame_height
    y_cm = round(koefficient_y * y_pixels, 3)
    return x_cm, y_cm


def pixels_to_meters(x_pixels, y_pixels, frame_width, frame_height):
    koefficient_x = 532.5 / 100 / frame_width
    x_meters = round(koefficient_x * x_pixels, 3)
    koefficient_y = 285 / 100 / frame_height
    y_meters = round(koefficient_y * y_pixels, 3)
    return x_meters, y_meters


def meters_to_pixels(x_meters, y_meters, frame_width, frame_height):
    koefficient_x = frame_width / 532.5 * 100
    x_pixels = round(koefficient_x * x_meters)
    koefficient_y = frame_height / 285 * 100
    y_pixels = round(koefficient_y * y_meters)
    return x_pixels, y_pixels


def convert_to_kf_frame(x_camera, y_camera, psi_camera):
    return x_camera, y_camera, psi_camera


def convert_to_camera_frame(x_kf, y_kf, psi_kf):
    return x_kf, y_kf, psi_kf


def main():
    index = 8

    video_path = f"files/Video_{index}.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Total frames: {total_frames}, FPS: {fps}")

    output_video_writer = create_video_writer(cap, f"files/Output_Video_{index}.mp4")

    pbar = tqdm(total=total_frames, desc="Processing Video")

    starting_x_camera = [0.0, 0.43, 0.23, 0.22, 0.0, 0.0, 2.51, 5.0]
    starting_y_camera = [0.0, 2.61, 0.33, 0.38, 0.0, 0.0, 1.12, 3.0]
    starting_psi_camera = [0.0, 0.0, -np.pi / 2, -np.pi / 2, 0.0, 0.0, 0.0, np.pi / 2]

    (
        starting_x_meters_kf,
        starting_y_meters_kf,
        starting_psi_meters_kf,
    ) = convert_to_kf_frame(
        starting_x_camera[index - 1],
        starting_y_camera[index - 1],
        starting_psi_camera[index - 1],
    )

    initial_X = State(
        X=starting_x_meters_kf,
        Y=starting_y_meters_kf,
        Psi=starting_psi_meters_kf,
        x_dot=0.0,
        y_dot=0.0,
        psi_dot=0.0,
    )
    print("initial X kf frame: ", initial_X)
    starting_U = ControlAction(0, 0)

    vehicle_info_rc_car_guess = VehicleParams(
        mass=0,
        moment_of_inertia=0,
        front_tire_stiffness_coefficient=0,
        rear_tire_stiffness_coefficient=0,
        front_axle_length=0.125,
        rear_axle_length=0.125,
        friction_coefficient=0,
    )

    model = KinematicBicycleModel(vehicle_info_rc_car_guess)

    sensor_fusion = PrerecordedSensorFusion(
        sensor_log_file_path=f"files/Log_{index}.log",
        model=model,
        vehicle_info=vehicle_info_rc_car_guess,
        sampling_time=1 / fps,
    )

    sensor_fusion.initialize(
        initial_state=initial_X,
        starting_control_input=starting_U,
        sampling_time=1 / fps,
    )

    sensors_fusion_sensors_only = PrerecordedSensorFusion(
        sensor_log_file_path=f"files/Log_{index}.log",
        model=model,
        vehicle_info=vehicle_info_rc_car_guess,
        sampling_time=1 / fps,
    )

    sensors_fusion_sensors_only.initialize(
        initial_state=initial_X,
        starting_control_input=starting_U,
        sampling_time=1 / fps,
    )

    kalman_filters_positions_meters: list[list[tuple[float, float]]] = [[], []]
    camera_positions = []
    cumulative_error = 0
    frame_count = 0
    error_rolling_window = [0] * 30

    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of video.")
            break

        # Get the current frame's timestamp in milliseconds
        current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        kf_state_meters = sensor_fusion.run_iteration(current_time_ms)
        put_state_on_frame(
            frame,
            kf_state_meters,
            starting_width=300,
            starting_height=100,
            height_offset=50,
            color=(0, 0, 255),
        )

        kf_state_meters_gps_only = (
            sensors_fusion_sensors_only.run_iteration_sensors_only(current_time_ms)
        )
        print(f"KF state: {kf_state_meters}")

        for kf_state_meters, starting_width, starting_height, color, index in [
            (kf_state_meters, 600, 100, (255, 255, 100), 0),
            (kf_state_meters_gps_only, 1350, 400, (0, 255, 0), 1),
        ]:
            kf_state_meters_in_camera_frame = convert_state_to_camera_frame(
                kf_state_meters
            )
            print(f"KF state in camera frame: {kf_state_meters_in_camera_frame}")

            put_state_on_frame(
                frame,
                kf_state_meters_in_camera_frame,
                starting_width=starting_width,
                starting_height=starting_height,
                height_offset=50,
                color=color,
            )

            kalman_filters_positions_meters[index].append(
                (kf_state_meters_in_camera_frame.X, kf_state_meters_in_camera_frame.Y)
            )

            draw_circles_for_positions(
                frame,
                kalman_filters_positions_meters[index],
                color,
                color,
                1920,
                1080,
                position_in_meters=True,
            )

        frame, camera_pixels_x, camera_pixels_y = process_frame(frame)

        if camera_pixels_x is not None and camera_pixels_y is not None:
            camera_positions.append((camera_pixels_x, camera_pixels_y))

            (_, _, camera_meters_x, camera_meters_y) = draw_circles_for_positions(
                frame,
                camera_positions,
                (250, 20, 250),
                (250, 20, 250),
                1920,
                1080,
                position_in_meters=False,
            )

            # current_error = np.sqrt(
            #     (kf_meters_x - camera_meters_x) ** 2
            #     + (kf_meters_y - camera_meters_y) ** 2
            # )
            #
            # # Update rolling error
            # error_rolling_window.pop(0)
            # error_rolling_window.append(current_error)
            #
            # rolling_error = np.mean(error_rolling_window)
            #
            # # Update cumulative error and frame count
            # frame_count += 1
            # cumulative_error += current_error
            #
            # # Calculate average error
            # avg_error = cumulative_error / frame_count
            #
            # draw_error(frame, avg_error, rolling_error, current_error)

        for kf_postions_meters in kalman_filters_positions_meters:
            if len(kf_postions_meters) > 50:
                kf_postions_meters.pop(0)
        if len(camera_positions) > 50:
            camera_positions.pop(0)

        cv2.namedWindow("Processed Frame", cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow("Processed Frame", 1280, 720)

        cv2.imshow("Processed Frame", frame)

        output_video_writer.write(frame)
        pbar.update(1)

        key = cv2.waitKey(20)
        if key & 0xFF == ord("q"):
            break
        elif key & 0xFF == ord("x"):
            cv2.waitKey(0)

    pbar.close()

    cap.release()
    output_video_writer.release()
    cv2.destroyAllWindows()


def convert_state_to_camera_frame(kf_state_meters):
    (
        kf_meters_x_in_camera_frame,
        kf_meters_y_in_camera_frame,
        kf_meters_pdi_in_camera_frame,
    ) = convert_to_camera_frame(
        kf_state_meters.X, kf_state_meters.Y, kf_state_meters.Psi
    )
    kf_state_meters_in_camera_frame = State(
        X=kf_meters_x_in_camera_frame,
        Y=kf_meters_y_in_camera_frame,
        Psi=kf_meters_pdi_in_camera_frame,
        x_dot=kf_state_meters.x_dot,
        y_dot=kf_state_meters.y_dot,
        psi_dot=kf_state_meters.psi_dot,
    )
    return kf_state_meters_in_camera_frame


if __name__ == "__main__":
    main()