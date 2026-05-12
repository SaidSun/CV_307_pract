import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str], cwd: Path | None = None) -> None:
    print(f"[RUN] {' '.join(command)}")
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def resolve_yolov5_dir(user_path: str) -> Path:
    path = Path(user_path).resolve()
    if (path / "train.py").exists():
        return path
    raise FileNotFoundError(
        f"Could not find train.py in YOLOv5 directory: {path}. "
        "Clone repo first: git clone https://github.com/ultralytics/yolov5.git"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YOLOv5 on prepared dataset (Windows GPU)."
    )
    parser.add_argument("--yolov5-dir", type=str, default="yolov5")
    parser.add_argument("--data", type=str, default="data/person.yaml")
    parser.add_argument("--weights", type=str, default="yolov5s.pt")
    parser.add_argument("--img", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", type=str, default="person_yolov5")
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--cache", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    yolov5_dir = resolve_yolov5_dir(args.yolov5_dir)
    data_yaml = (project_root / args.data).resolve()

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset yaml not found: {data_yaml}. "
            "Prepare dataset and create data/person.yaml first."
        )

    if os.name == "nt":
        pip_cmd = [sys.executable, "-m", "pip"]
    else:
        pip_cmd = ["pip"]

    run_command(pip_cmd + ["install", "-r", "requirements.txt"], cwd=yolov5_dir)

    train_cmd = [
        sys.executable,
        "train.py",
        "--img",
        str(args.img),
        "--batch",
        str(args.batch),
        "--epochs",
        str(args.epochs),
        "--data",
        str(data_yaml),
        "--weights",
        args.weights,
        "--device",
        args.device,
        "--workers",
        str(args.workers),
        "--project",
        args.project,
        "--name",
        args.name,
    ]

    if args.cache:
        train_cmd.append("--cache")

    run_command(train_cmd, cwd=yolov5_dir)

    print("--------------------------------")
    print("Training started/completed successfully.")
    print(f"Check artifacts in: {yolov5_dir / args.project / args.name}")
    print("--------------------------------")


if __name__ == "__main__":
    main()
