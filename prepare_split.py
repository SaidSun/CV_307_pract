#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare YOLO train/test split from original and augmented samples."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("CV307_detector_jpeg"),
        help="Source folder with original images and labels.",
    )
    parser.add_argument(
        "--aug",
        type=Path,
        default=Path("dataset_augmented/raw"),
        help="Augmentation folder from augment_rotate.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("dataset"),
        help="Output YOLO dataset directory.",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.8,
        help="Train split ratio (0.0..1.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split.",
    )
    return parser.parse_args()


def read_classes(classes_path: Path) -> List[str]:
    classes = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines()]
    return [c for c in classes if c]


def gather_pairs_flat(base_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for img_path in sorted(base_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = base_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            pairs.append((img_path, label_path))
    return pairs


def gather_pairs_aug(aug_dir: Path) -> List[Tuple[Path, Path]]:
    images_dir = aug_dir / "images"
    labels_dir = aug_dir / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        return []

    pairs: List[Tuple[Path, Path]] = []
    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            pairs.append((img_path, label_path))
    return pairs


def reset_output_dirs(out_dir: Path) -> None:
    for rel in (
        "images/train",
        "images/test",
        "labels/train",
        "labels/test",
    ):
        p = out_dir / rel
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)


def copy_pairs(
    pairs: Iterable[Tuple[Path, Path]],
    image_dest: Path,
    label_dest: Path,
) -> None:
    for img_src, lbl_src in pairs:
        shutil.copy2(img_src, image_dest / img_src.name)
        shutil.copy2(lbl_src, label_dest / lbl_src.name)


def write_dataset_yaml(out_dir: Path, class_names: Sequence[str]) -> Path:
    yaml_path = out_dir / "dataset.yaml"
    names_items = ", ".join(repr(name) for name in class_names)
    content = (
        f"path: {out_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/test\n"
        "test: images/test\n"
        f"nc: {len(class_names)}\n"
        f"names: [{names_items}]\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def main() -> None:
    args = parse_args()
    if not 0.0 < args.split < 1.0:
        raise ValueError("--split must be in range (0, 1)")

    src_dir = args.src.resolve()
    aug_dir = args.aug.resolve()
    out_dir = args.out.resolve()

    classes_path = src_dir / "classes.txt"
    if not classes_path.exists():
        raise FileNotFoundError(f"classes.txt not found in {src_dir}")
    class_names = read_classes(classes_path)
    if not class_names:
        raise ValueError("No classes found in classes.txt")

    src_pairs = gather_pairs_flat(src_dir)
    aug_pairs = gather_pairs_aug(aug_dir)
    all_pairs = src_pairs + aug_pairs
    if not all_pairs:
        raise ValueError("No valid image/label pairs found.")

    random.seed(args.seed)
    random.shuffle(all_pairs)

    train_count = max(1, int(len(all_pairs) * args.split))
    if train_count >= len(all_pairs):
        train_count = len(all_pairs) - 1

    train_pairs = all_pairs[:train_count]
    test_pairs = all_pairs[train_count:]

    reset_output_dirs(out_dir)

    copy_pairs(train_pairs, out_dir / "images/train", out_dir / "labels/train")
    copy_pairs(test_pairs, out_dir / "images/test", out_dir / "labels/test")
    yaml_path = write_dataset_yaml(out_dir, class_names)

    print(f"Original paired samples: {len(src_pairs)}")
    print(f"Augmented paired samples: {len(aug_pairs)}")
    print(f"Total paired samples: {len(all_pairs)}")
    print(f"Train samples: {len(train_pairs)}")
    print(f"Test samples: {len(test_pairs)}")
    print(f"Classes: {len(class_names)}")
    print(f"Dataset YAML: {yaml_path}")


if __name__ == "__main__":
    main()
