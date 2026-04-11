# CV307 YOLO Pipeline

## 1) Install dependencies

```bash
pip install ultralytics pillow
```

## 2) Generate rotated augmentations (90/180/270)

```bash
python augment_rotate.py --src "CV307_detector_jpeg" --out "dataset_augmented/raw"
```

Outputs:
- `dataset_augmented/raw/images`
- `dataset_augmented/raw/labels`

## 3) Build train/test dataset and YAML (80/20)

```bash
python prepare_split.py --src "CV307_detector_jpeg" --aug "dataset_augmented/raw" --out "dataset" --split 0.8 --seed 42
```

Outputs:
- `dataset/images/train`, `dataset/images/test`
- `dataset/labels/train`, `dataset/labels/test`
- `dataset/dataset.yaml`

## 4) Train small YOLO model and print metrics

```bash
python train_yolo.py --data "dataset/dataset.yaml" --epochs 50 --imgsz 640 --batch 16 --device cpu
```

Printed metrics:
- Train: `precision`, `recall`, `mAP50`, `mAP50-95`
- Test: `precision`, `recall`, `mAP50`, `mAP50-95`

## Validation checklist

- Check random augmented images and confirm bboxes still match objects.
- Ensure image and label counts match in both `train` and `test`.
- Run with same `--seed` and verify split reproducibility.
- Watch for overfitting: train metrics much higher than test metrics.
