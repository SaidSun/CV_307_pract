import os
os.environ["YOLOv5_AUTOINSTALL"] = "False"

import cv2
import torch
import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def clip_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(float(x1), w - 1))
    y1 = max(0, min(float(y1), h - 1))
    x2 = max(0, min(float(x2), w - 1))
    y2 = max(0, min(float(y2), h - 1))
    return x1, y1, x2, y2


def box_to_yolo(x1, y1, x2, y2, img_w, img_h):
    x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, img_w, img_h)

    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 1 or box_h <= 1:
        return None

    x_center = x1 + box_w / 2
    y_center = y1 + box_h / 2

    x_center /= img_w
    y_center /= img_h
    box_w /= img_w
    box_h /= img_h

    return x_center, y_center, box_w, box_h


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_image_for_label(obj_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = obj_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def validate_pairs(obj_dir: Path) -> Tuple[List[Tuple[Path, Path]], List[str]]:
    labels = sorted(obj_dir.glob("*.txt"))
    images_by_stem: Dict[str, Path] = {p.stem: p for ext in IMAGE_EXTENSIONS for p in obj_dir.glob(f"*{ext}")}
    pairs: List[Tuple[Path, Path]] = []
    warnings: List[str] = []

    label_stems = set()
    for label_path in labels:
        label_stem = label_path.stem
        label_stems.add(label_stem)
        image_path = find_image_for_label(obj_dir, label_stem)
        if image_path is None:
            warnings.append(f"Label without image: {label_path.name}")
            continue
        pairs.append((image_path, label_path))

    for image_stem, image_path in images_by_stem.items():
        if image_stem not in label_stems:
            warnings.append(f"Image without label: {image_path.name}")

    return pairs, warnings


def split_pairs(
    pairs: List[Tuple[Path, Path]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[Tuple[Path, Path]]]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    shuffled = pairs[:]
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:n_train + n_val + n_test],
    }


def ensure_yolov5_dirs(root_dir: Path) -> Dict[str, Dict[str, Path]]:
    paths: Dict[str, Dict[str, Path]] = {}
    for split in ("train", "val", "test"):
        img_dir = root_dir / "images" / split
        lbl_dir = root_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        paths[split] = {"images": img_dir, "labels": lbl_dir}
    return paths


def prepare_yolov5_dataset(
    obj_dir: Path,
    out_root: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    clear_output: bool,
) -> None:
    if not obj_dir.exists():
        raise FileNotFoundError(f"Source folder does not exist: {obj_dir}")

    pairs, warnings = validate_pairs(obj_dir)
    if warnings:
        print("Warnings during dataset validation:")
        for warning in warnings:
            print(f"  - {warning}")

    if not pairs:
        raise RuntimeError(
            "No valid image/label pairs found. Ensure images are restored in obj/."
        )

    if clear_output and out_root.exists():
        shutil.rmtree(out_root)

    split_map = split_pairs(pairs, train_ratio, val_ratio, test_ratio, seed)
    dir_map = ensure_yolov5_dirs(out_root)

    for split_name, split_pairs_list in split_map.items():
        img_out_dir = dir_map[split_name]["images"]
        lbl_out_dir = dir_map[split_name]["labels"]

        for image_path, label_path in split_pairs_list:
            shutil.copy2(image_path, img_out_dir / image_path.name)
            shutil.copy2(label_path, lbl_out_dir / label_path.name)

    print("--------------------------------")
    print(f"Prepared YOLOv5 dataset at: {out_root.resolve()}")
    print(f"Total valid pairs: {len(pairs)}")
    for split_name in ("train", "val", "test"):
        print(f"{split_name}: {len(split_map[split_name])}")
    print("--------------------------------")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="generate",
        choices=("generate", "prepare", "all"),
        help="generate: video->obj, prepare: obj->yolov5 split, all: both",
    )
    parser.add_argument("--video", type=str, default="8.avi")
    parser.add_argument("--out", type=str, default="dataset_person")
    parser.add_argument("--every", type=int, default=1, help="Сохранять каждый N-й кадр")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--model", type=str, default="yolov5m", help="yolov5s / yolov5m / yolov5l")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = обработать всё видео")
    parser.add_argument("--split-train", type=float, default=0.7)
    parser.add_argument("--split-val", type=float, default=0.2)
    parser.add_argument("--split-test", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--clear-prepared", action="store_true")
    parser.add_argument(
        "--prepared-dir",
        type=str,
        default="dataset_person_yolov5",
        help="Output folder with images/labels/{train,val,test}",
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.out)
    obj_dir = output_dir / "obj"
    preview_dir = output_dir / "preview"

    if args.mode in ("generate", "all"):
        obj_dir.mkdir(parents=True, exist_ok=True)

        if args.preview:
            preview_dir.mkdir(parents=True, exist_ok=True)

        model = torch.hub.load("ultralytics/yolov5", args.model, pretrained=True)
        model.to(args.device)
        model.eval()

        model.conf = args.conf
        model.iou = 0.45

        person_class_id = 0

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {video_path}")

        frame_idx = 0
        saved_idx = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break

            if frame_idx % args.every != 0:
                frame_idx += 1
                continue

            img_h, img_w = frame.shape[:2]

            results = model(frame)
            detections = results.xyxy[0].detach().cpu().numpy()

            image_name = f"frame_{saved_idx:06d}.jpg"
            label_name = f"frame_{saved_idx:06d}.txt"

            image_path = obj_dir / image_name
            label_path = obj_dir / label_name

            label_lines = []

            preview_frame = frame.copy()

            for det in detections:
                x1, y1, x2, y2, conf, cls = det

                if int(cls) != person_class_id:
                    continue

                if float(conf) < args.conf:
                    continue

                yolo_box = box_to_yolo(x1, y1, x2, y2, img_w, img_h)

                if yolo_box is None:
                    continue

                xc, yc, bw, bh = yolo_box

                # YOLO txt format:
                # class_id x_center y_center width height
                label_lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

                if args.preview:
                    x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                    cv2.rectangle(preview_frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
                    cv2.putText(
                        preview_frame,
                        f"person {float(conf):.2f}",
                        (x1i, max(20, y1i - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

            cv2.imwrite(str(image_path), frame)

            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(label_lines))

            if args.preview:
                preview_path = preview_dir / image_name
                cv2.imwrite(str(preview_path), preview_frame)

            print(f"[OK] {image_name}: найдено людей = {len(label_lines)}")

            saved_idx += 1
            frame_idx += 1

        cap.release()

        with open(output_dir / "obj.names", "w", encoding="utf-8") as f:
            f.write("person\n")

        print("--------------------------------")
        print(f"Готово. Сохранено кадров: {saved_idx}")
        print(f"Датасет: {output_dir.resolve()}")
        print("--------------------------------")

    if args.mode in ("prepare", "all"):
        prepare_yolov5_dataset(
            obj_dir=output_dir / "obj",
            out_root=Path(args.prepared_dir),
            train_ratio=args.split_train,
            val_ratio=args.split_val,
            test_ratio=args.split_test,
            seed=args.split_seed,
            clear_output=args.clear_prepared,
        )


if __name__ == "__main__":
    main()