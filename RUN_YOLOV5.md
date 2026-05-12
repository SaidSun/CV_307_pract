# YOLOv5 Training (Windows GPU)

## 1) Restore images
- Put all images back into `dataset_person/obj/` so each `frame_xxxxxx.txt` has matching `frame_xxxxxx.jpg` (or `.png`, `.jpeg`, `.bmp`, `.webp`).

## 2) Prepare split 70/20/10
```bash
python yolo_datasetMaker.py --mode prepare --out dataset_person --prepared-dir dataset_person_yolov5 --split-train 0.7 --split-val 0.2 --split-test 0.1 --split-seed 42 --clear-prepared
```

## 3) Clone YOLOv5
```bash
git clone https://github.com/ultralytics/yolov5.git
```

## 4) Train model
```bash
python train_yolov5_windows.py --yolov5-dir yolov5 --data data/person.yaml --weights yolov5s.pt --img 640 --batch 8 --epochs 100 --device 0
```

## 5) Verify artifacts
- Expected output directory: `yolov5/runs/train/person_yolov5/`
- Key files:
  - `weights/best.pt`
  - `weights/last.pt`
  - `results.csv`
