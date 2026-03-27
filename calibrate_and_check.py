import argparse
import glob
import json
import os
import uuid
from datetime import datetime

import cv2
import numpy as np


CHESS_AXIS = np.float32([[0.1, 0, 0], [0, 0.1, 0], [0, 0, -0.1]]).reshape(-1, 3)
DEFAULT_TARGET_FRAMES = 12
MIN_CALIBRATION_FRAMES = 8


def parse_source(source):
    if isinstance(source, str) and len(source) == 1 and source.isdigit():
        return int(source)
    return source


def build_chessboard_model(pattern_size=(9, 6), square_size=0.025):
    objp = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
    return objp


def find_chessboard(img, pattern_size, criteria):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_size = gray.shape
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    cv2.drawChessboardCorners(img, pattern_size, corners, ret)
    if ret and len(corners):
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return True, corners, image_size
    return False, None, image_size


def calc_total_error(objpoints, imgpoints, camera_matrix, dist_coeffs, rvecs, tvecs):
    mean_error = 0.0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    total_error = mean_error / len(objpoints)
    print("Total error:", total_error)
    return total_error


def draw_capture_hud(img, frame_counter, target_frames, found):
    hud = img.copy()
    cv2.rectangle(hud, (10, 10), (360, 126), (20, 20, 20), -1)
    cv2.addWeighted(hud, 0.55, img, 0.45, 0, img)
    status_text = "BOARD: LOCKED" if found else "BOARD: SEARCHING"
    status_color = (0, 220, 120) if found else (80, 180, 255)
    cv2.putText(img, "CALIBRATION MODE", (20, 34), cv2.FONT_HERSHEY_DUPLEX, 0.65, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(img, status_text, (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2, cv2.LINE_AA)
    cv2.putText(img, "Enter - capture frame", (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(img, "Q/Esc - finish", (20, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)
    progress = min(frame_counter / float(max(target_frames, 1)), 1.0)
    bar_x, bar_y, bar_w, bar_h = 20, 112, 320, 10
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 70, 70), 1)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), (0, 180, 255), -1)
    cv2.putText(img, "%d/%d" % (frame_counter, target_frames), (275, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)


def draw_corner_style(img, corners):
    for idx, point in enumerate(corners.reshape(-1, 2)):
        x, y = int(point[0]), int(point[1])
        color = (255, 120, 0) if idx % 2 == 0 else (80, 220, 255)
        cv2.circle(img, (x, y), 3, color, -1, lineType=cv2.LINE_AA)


def draw_pose_axis(img, corners, imgpts):
    corner = tuple(int(coord) for coord in corners[0].ravel())
    p1 = tuple(int(coord) for coord in imgpts[0].ravel())
    p2 = tuple(int(coord) for coord in imgpts[1].ravel())
    p3 = tuple(int(coord) for coord in imgpts[2].ravel())
    img = cv2.line(img, corner, p1, (255, 0, 0), 5)
    img = cv2.line(img, corner, p2, (0, 255, 0), 5)
    img = cv2.line(img, corner, p3, (0, 0, 255), 5)
    cv2.circle(img, corner, 7, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    return img


def draw_pose_label(img, position, distance):
    hud = img.copy()
    cv2.rectangle(hud, (10, 10), (500, 130), (24, 24, 24), -1)
    cv2.addWeighted(hud, 0.55, img, 0.45, 0, img)
    cv2.putText(img, "LIVE CHECK", (20, 33), cv2.FONT_HERSHEY_DUPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(img, "Axis colors:", (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (225, 225, 225), 1, cv2.LINE_AA)
    cv2.putText(img, "X", (140, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "Y", (168, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "Z", (196, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "(same as board axes)", (225, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)

    cv2.putText(img, "x: %.4f m" % position[0], (20, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "y: %.4f m" % position[1], (180, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "z: %.4f m" % position[2], (340, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(
        img,
        "Distance |R| from board origin: %.4f m" % distance,
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (60, 220, 255),
        2,
        cv2.LINE_AA,
    )


def run_video_capture_for_calibration(source, pattern_size, objp, criteria, save_images_path, all_frames, target_frames):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Unable to open source for calibration:", source)
        return None, None, None

    objpoints = []
    imgpoints = []
    frame_counter = 0
    image_size = None

    while True:
        save_flag = bool(all_frames)
        flag, img = cap.read()
        if not flag:
            break
        clear_img = img.copy()
        found, corners, image_size = find_chessboard(img, pattern_size, criteria)
        if found and corners is not None:
            draw_corner_style(img, corners)

        display = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
        draw_capture_hud(display, frame_counter, target_frames, found)
        cv2.imshow("Calibration Studio", display)
        ch = cv2.waitKey(1)

        if ch == 13:
            if not all_frames:
                save_flag = True
            if found and corners is not None:
                objpoints.append(objp)
                imgpoints.append(corners)
                frame_counter += 1

        if ch == ord("q") or ch == 27:
            break

        if all_frames and found and corners is not None:
            objpoints.append(objp)
            imgpoints.append(corners)
            frame_counter += 1

        if frame_counter >= target_frames:
            print("Collected target number of frames:", target_frames)
            break

        if save_flag and save_images_path is not None:
            name = "%s.jpeg" % frame_counter
            cv2.imwrite(os.path.join(save_images_path, name), clear_img)

    cv2.destroyAllWindows()
    cap.release()
    return objpoints, imgpoints, image_size


def run_images_capture_for_calibration(images_path, pattern_size, objp, criteria, target_frames):
    objpoints = []
    imgpoints = []
    image_size = None
    images = glob.glob(os.path.join(images_path, "*.jpg")) + glob.glob(os.path.join(images_path, "*.jpeg"))

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
        found, corners, image_size = find_chessboard(img, pattern_size, criteria)
        if found and corners is not None:
            draw_corner_style(img, corners)
            objpoints.append(objp)
            imgpoints.append(corners)
        display = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
        draw_capture_hud(display, len(objpoints), target_frames, found)
        cv2.imshow("Calibration Studio", display)
        ch = cv2.waitKey(1)
        if ch == ord("q") or ch == 27:
            break
        if len(objpoints) >= target_frames:
            print("Collected target number of frames:", target_frames)
            break

    cv2.destroyAllWindows()
    return objpoints, imgpoints, image_size


def calibrate_camera(objpoints, imgpoints, image_size, source, source_type):
    if not objpoints or not imgpoints or image_size is None or len(objpoints) < MIN_CALIBRATION_FRAMES:
        print("Not enough valid frames for calibration. Need at least %d, got %d." % (MIN_CALIBRATION_FRAMES, len(objpoints) if objpoints else 0))
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size[::-1],
        None,
        None,
        criteria,
        flags=cv2.CALIB_FIX_PRINCIPAL_POINT,
    )
    if not ret:
        print("Calibration failed.")
        return None

    resolution = {"h": image_size[0], "w": image_size[1]}
    optimal_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        (resolution["w"], resolution["h"]),
        1,
        (resolution["w"], resolution["h"]),
    )
    total_error = calc_total_error(objpoints, imgpoints, camera_matrix, dist_coeffs, rvecs, tvecs)
    print("-" * 20)
    print("Camera Matrix:", camera_matrix)
    print("Distortion Coefficients:", dist_coeffs)
    print("-" * 20)

    camera = {
        "id": str(uuid.uuid4()),
        "type": source_type,
        "calibration_source": source,
        "camera_matrix": camera_matrix.tolist(),
        "optimal_camera_matrix": optimal_camera_matrix.tolist(),
        "roi": roi,
        "distortion": dist_coeffs.tolist(),
        "rvecs": [vec.tolist() for vec in rvecs],
        "tvecs": [vec.tolist() for vec in tvecs],
        "resolution": resolution,
        "total_error": total_error,
    }
    return camera


def save_calibration(path, data):
    if data is None:
        print("Calibration data is None.")
        return None
    if not os.path.exists(path):
        os.makedirs(path)
        print("Create directory:", path)
    filename = "calibration__%s.json" % datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    full_path = os.path.join(path, filename)
    with open(full_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file)
    print("Calibration data saved in:", full_path)
    return full_path


def run_live_check(source, pattern_size, square_size, camera_matrix, dist_coeffs):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Unable to open source for live check:", source)
        return 1

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.001)
    objp = build_chessboard_model(pattern_size=pattern_size, square_size=square_size)
    print("Live check started. Press Esc or Q to exit.")

    while True:
        flag, img = cap.read()
        if not flag:
            break

        found, corners, _ = find_chessboard(img, pattern_size, criteria)
        if found and corners is not None:
            draw_corner_style(img, corners)
            ret, rvecs, tvecs = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
            if ret:
                rot_m = cv2.Rodrigues(rvecs)[0]
                camera_world_position = -np.matrix(rot_m).T * np.matrix(tvecs)
                camera_world_position = np.ravel(camera_world_position)
                camera_distance = float(np.linalg.norm(camera_world_position))
                imgpts, _ = cv2.projectPoints(CHESS_AXIS, rvecs, tvecs, camera_matrix, dist_coeffs)
                draw_pose_axis(img, corners, imgpts)
                draw_pose_label(img, camera_world_position, camera_distance)

        display = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
        cv2.imshow("Live Pose Inspector", display)
        ch = cv2.waitKey(1)
        if ch == ord("q") or ch == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
    return 0


def main():
    parser = argparse.ArgumentParser(description="Standalone chessboard calibration and live validation")
    parser.add_argument("--source", "-s", required=True, help="Camera source, video path, or images folder")
    parser.add_argument("--source_type", "-t", default="rgb", help="Camera type metadata")
    parser.add_argument("--board_type", "-b", default="chess", help="Only chess is supported in this script")
    parser.add_argument("--save_path", default="./out/", help="Where calibration json will be saved")
    parser.add_argument("--save_images", default=None, help="Where captured calibration frames will be saved")
    parser.add_argument("--all_frames", "-a", action="store_true", help="Use all valid video frames for calibration")
    parser.add_argument("--target_frames", type=int, default=DEFAULT_TARGET_FRAMES, help="Target number of valid frames to collect")
    parser.add_argument("--pattern_cols", type=int, default=9, help="Chessboard inner corners by columns")
    parser.add_argument("--pattern_rows", type=int, default=6, help="Chessboard inner corners by rows")
    parser.add_argument("--square_size", type=float, default=0.025, help="Chessboard square size in meters")
    args = parser.parse_args()

    if args.board_type.lower() != "chess":
        print("Only chess board is implemented in standalone script.")
        return 1

    source_value = parse_source(args.source)
    pattern_size = (args.pattern_cols, args.pattern_rows)
    if args.save_images and not os.path.exists(args.save_images):
        os.makedirs(args.save_images)
        print("Create directory:", args.save_images)

    objp = build_chessboard_model(pattern_size=pattern_size, square_size=args.square_size)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.001)

    target_frames = max(MIN_CALIBRATION_FRAMES, args.target_frames)
    print("Target frames:", target_frames)
    print("Minimum required frames:", MIN_CALIBRATION_FRAMES)

    if isinstance(source_value, str) and os.path.isdir(source_value):
        objpoints, imgpoints, image_size = run_images_capture_for_calibration(
            source_value, pattern_size, objp, criteria, target_frames
        )
    else:
        objpoints, imgpoints, image_size = run_video_capture_for_calibration(
            source_value, pattern_size, objp, criteria, args.save_images, args.all_frames, target_frames
        )

    camera = calibrate_camera(objpoints, imgpoints, image_size, args.source, args.source_type)
    if camera is None:
        return 1

    calibration_data = {
        "cameras": [camera],
        "board": {
            "type": "chess",
            "pattern_size": pattern_size,
            "square_size": args.square_size,
        },
    }
    save_calibration(args.save_path, calibration_data)

    camera_matrix = np.array(camera["camera_matrix"])
    dist_coeffs = np.array(camera["distortion"])
    return run_live_check(source_value, pattern_size, args.square_size, camera_matrix, dist_coeffs)


if __name__ == "__main__":
    raise SystemExit(main())
