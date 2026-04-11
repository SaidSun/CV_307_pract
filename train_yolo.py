#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8n and report train/test metrics.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("dataset/dataset.yaml"),
        help="Path to dataset yaml.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of train epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training/validation.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Device for training, e.g. "cpu", "0".',
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Output project directory for Ultralytics runs.",
    )
    parser.add_argument("--name", type=str, default="cv307_yolov8n", help="Run name.")
    return parser.parse_args()


def extract_metrics(metrics_obj: Any) -> Dict[str, float]:
    box = getattr(metrics_obj, "box", None)
    if box is None:
        return {}
    return {
        "precision": float(getattr(box, "mp", 0.0)),
        "recall": float(getattr(box, "mr", 0.0)),
        "mAP50": float(getattr(box, "map50", 0.0)),
        "mAP50-95": float(getattr(box, "map", 0.0)),
    }


def print_metrics(title: str, values: Dict[str, float]) -> None:
    print(f"\n{title}")
    if not values:
        print("  Metrics are unavailable.")
        return
    print(f"  precision: {values['precision']:.4f}")
    print(f"  recall:    {values['recall']:.4f}")
    print(f"  mAP50:     {values['mAP50']:.4f}")
    print(f"  mAP50-95:  {values['mAP50-95']:.4f}")


def main() -> None:
    args = parse_args()
    data_yaml = args.data.resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")

    model = YOLO("yolov8n.pt")
    train_result = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )

    best_path = Path(train_result.save_dir) / "weights" / "best.pt"
    model_best = YOLO(str(best_path)) if best_path.exists() else model

    train_metrics = model_best.val(data=str(data_yaml), split="train", imgsz=args.imgsz, device=args.device)
    test_metrics = model_best.val(data=str(data_yaml), split="test", imgsz=args.imgsz, device=args.device)

    print_metrics("Train metrics:", extract_metrics(train_metrics))
    print_metrics("Test metrics:", extract_metrics(test_metrics))
    print(f"\nRun directory: {train_result.save_dir}")
    print(f"Best weights: {best_path if best_path.exists() else 'not found'}")


if __name__ == "__main__":
    main()
