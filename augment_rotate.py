#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
ROTATIONS = (90, 180, 270)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create rotated image augmentations and rotated YOLO labels."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("CV307_detector_jpeg"),
        help="Source folder with images and YOLO txt labels.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("dataset_augmented/raw"),
        help="Output folder for augmented images/labels.",
    )
    return parser.parse_args()


def list_images(src_dir: Path) -> List[Path]:
    return sorted(
        p
        for p in src_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def read_yolo_label(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    records: List[Tuple[int, float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO row in {label_path}: {line!r}")
        cls_id = int(parts[0])
        x, y, w, h = map(float, parts[1:])
        records.append((cls_id, x, y, w, h))
    return records


def rotate_bbox_yolo(
    x: float,
    y: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
    angle: int,
) -> Tuple[float, float, float, float]:
    cx = x * img_w
    cy = y * img_h
    bw = w * img_w
    bh = h * img_h

    if angle == 90:
        new_w, new_h = img_h, img_w
        cx_new = cy
        cy_new = img_w - cx
        bw_new = bh
        bh_new = bw
    elif angle == 180:
        new_w, new_h = img_w, img_h
        cx_new = img_w - cx
        cy_new = img_h - cy
        bw_new = bw
        bh_new = bh
    elif angle == 270:
        new_w, new_h = img_h, img_w
        cx_new = img_h - cy
        cy_new = cx
        bw_new = bh
        bh_new = bw
    else:
        raise ValueError(f"Unsupported angle: {angle}")

    x_new = max(0.0, min(1.0, cx_new / new_w))
    y_new = max(0.0, min(1.0, cy_new / new_h))
    w_new = max(0.0, min(1.0, bw_new / new_w))
    h_new = max(0.0, min(1.0, bh_new / new_h))
    return x_new, y_new, w_new, h_new


def format_records(
    records: Iterable[Tuple[int, float, float, float, float]]
) -> str:
    lines = []
    for cls_id, x, y, w, h in records:
        lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def rotated_image(image: Image.Image, angle: int) -> Image.Image:
    if angle == 90:
        return image.transpose(Image.Transpose.ROTATE_90)
    if angle == 180:
        return image.transpose(Image.Transpose.ROTATE_180)
    if angle == 270:
        return image.transpose(Image.Transpose.ROTATE_270)
    raise ValueError(f"Unsupported angle: {angle}")


def main() -> None:
    args = parse_args()
    src_dir = args.src.resolve()
    out_dir = args.out.resolve()
    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(src_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {src_dir}")

    skipped = 0
    generated = 0
    for img_path in image_paths:
        label_path = src_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            skipped += 1
            continue

        with Image.open(img_path) as img:
            width, height = img.size
            labels = read_yolo_label(label_path)

            for angle in ROTATIONS:
                out_name = f"{img_path.stem}_rot{angle}{img_path.suffix.lower()}"
                out_img_path = out_images / out_name
                out_lbl_path = out_labels / f"{img_path.stem}_rot{angle}.txt"

                img_rot = rotated_image(img, angle)
                img_rot.save(out_img_path)

                rotated_labels = []
                for cls_id, x, y, w, h in labels:
                    rx, ry, rw, rh = rotate_bbox_yolo(
                        x, y, w, h, width, height, angle
                    )
                    rotated_labels.append((cls_id, rx, ry, rw, rh))
                out_lbl_path.write_text(format_records(rotated_labels), encoding="utf-8")
                generated += 1

    print(f"Source images: {len(image_paths)}")
    print(f"Generated augmented samples: {generated}")
    print(f"Skipped (missing label): {skipped}")
    print(f"Augmented images dir: {out_images}")
    print(f"Augmented labels dir: {out_labels}")


if __name__ == "__main__":
    main()
