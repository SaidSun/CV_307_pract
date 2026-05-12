import cv2
import torch
import math
import argparse
from pathlib import Path
from deep_sort_realtime.deepsort_tracker import DeepSort

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov5/runs/train/person_yolov5/weights/best.pt",
        help="Path to trained YOLOv5 weights",
    )
    parser.add_argument("--video", type=str, default="8.avi")
    parser.add_argument("--output", type=str, default="output.avi")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def signed_distance_to_line(point, line_start, line_end):
    """
    Возвращает signed distance от точки до линии в пикселях.
    Для горизонтальной линии:
    значение > 0 — точка с одной стороны,
    значение < 0 — с другой.
    """
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end

    dx = x2 - x1
    dy = y2 - y1

    length = math.sqrt(dx * dx + dy * dy)
    if length == 0:
        return 0

    return (dx * (py - y1) - dy * (px - x1)) / length


def main():
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    model = torch.hub.load("ultralytics/yolov5", "custom", path=str(weights_path))
    model.to(args.device)
    model.conf = args.conf
    model.iou = args.iou

    person_class_id = 0

    tracker = DeepSort(
        max_age=50,
        n_init=1
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {args.video}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25

    out = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (w, h)
    )

    line = [(0, h // 2), (w - 1, h // 2)]
    count = 0
    crossed_ids = set()
    previous_positions = {}
    line_tolerance = 8
    window_name = "YOLO + DeepSORT + Counting"

    last_frame = None

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()

        person_detections = []

        for det in detections:
            x1, y1, x2, y2, conf, cls = det

            if int(cls) == person_class_id and conf > args.conf:
                bbox = [
                    float(x1),
                    float(y1),
                    float(x2 - x1),
                    float(y2 - y1)
                ]
                person_detections.append((bbox, float(conf), "person"))

        tracks = tracker.update_tracks(person_detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            foot_x = int((l + r) / 2)
            foot_y = int(b)
            current_point = (foot_x, foot_y)

            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {track_id}",
                (l, t - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )
            cv2.circle(frame, current_point, 5, (255, 0, 0), -1)

            if track_id in previous_positions and track_id not in crossed_ids:
                previous_point = previous_positions[track_id]
                prev_dist = signed_distance_to_line(previous_point, line[0], line[1])
                curr_dist = signed_distance_to_line(current_point, line[0], line[1])
                crossed_line = prev_dist * curr_dist < 0
                touched_line = (
                    abs(curr_dist) <= line_tolerance
                    and abs(prev_dist) > line_tolerance
                )

                if crossed_line or touched_line:
                    count += 1
                    crossed_ids.add(track_id)

            previous_positions[track_id] = current_point

        cv2.line(frame, line[0], line[1], (0, 0, 255), 3)
        cv2.putText(
            frame,
            f"Count: {count}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        out.write(frame)
        last_frame = frame.copy()
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    if last_frame is not None:
        final_frame = last_frame.copy()
        cv2.putText(
            final_frame,
            f"Final count: {count}",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )
        pause_frames = int(fps * 2)
        for _ in range(pause_frames):
            out.write(final_frame)
        cv2.imshow(window_name, final_frame)
        cv2.waitKey(2000)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("--------------------------------")
    print(f"Итоговое количество людей: {count}")
    print("--------------------------------")


if __name__ == "__main__":
    main()