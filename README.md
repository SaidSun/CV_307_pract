# PR5_CV307

Проект для детекции и подсчета людей на видео с использованием `YOLOv5` и `DeepSORT`.

## Что делает проект

- подготавливает датасет в формате YOLO (из видео или из уже готовых `image+label`),
- дообучает `YOLOv5` на своем датасете,
- запускает детекцию + трекинг + подсчет пересечений линии в видео.

## Структура проекта

### Основные скрипты

- `yolo_datasetMaker.py`  
  Подготовка датасета:
  - `--mode generate` - генерирует `dataset_person/obj` из видео;
  - `--mode prepare` - валидирует пары `image<->label` и раскладывает в `images/labels + train/val/test`;
  - `--mode all` - выполняет оба шага подряд.

- `train_yolov5_windows.py`  
  Скрипт обучения `YOLOv5` на Windows GPU:
  - устанавливает зависимости YOLOv5,
  - запускает `train.py` с заданными параметрами (`img`, `batch`, `epochs`, `weights`, `device`).

- `YoloDsortDetection.py`  
  Инференс на видео:
  - загружает обученные веса (`best.pt` по умолчанию),
  - детектирует людей,
  - трекает объекты через `DeepSORT`,
  - считает пересечения контрольной линии.

### Данные и конфиги

- `dataset_person/obj/`  
  Исходные изображения и YOLO-лейблы (`.txt`) до разбиения на выборки.

- `dataset_person/obj.names`  
  Список классов датасета (в текущем проекте класс `person`).

- `dataset_person_yolov5/`  
  Подготовленный датасет для YOLOv5:
  - `images/train`, `images/val`, `images/test`
  - `labels/train`, `labels/val`, `labels/test`

- `data/person.yaml`  
  Конфиг датасета для обучения YOLOv5 (пути к train/val/test и список классов).

### Результаты обучения

- `yolov5/`  
  Локальная копия репозитория Ultralytics YOLOv5 (код обучения/инференса).

- `yolov5/runs/train/person_yolov5/`  
  Артефакты последнего обучения:
  - `weights/best.pt` - лучшая модель,
  - `weights/last.pt` - последняя эпоха,
  - `results.csv`, `hyp.yaml`, `opt.yaml` - метрики и параметры запуска.

### Вспомогательные файлы

- `RUN_YOLOV5.md`  
  Короткая инструкция по запуску подготовки данных и обучения.

- `.ipynb_checkpoints/`  
  Автосохранения Jupyter/IDE (не основной рабочий код).

## Базовый сценарий работы

1. Восстановить изображения в `dataset_person/obj` (чтобы у каждого `.txt` была соответствующая картинка).
2. Подготовить датасет под YOLOv5 (`70/20/10`):
   - `python yolo_datasetMaker.py --mode prepare --out dataset_person --prepared-dir dataset_person_yolov5 --split-train 0.7 --split-val 0.2 --split-test 0.1 --split-seed 42 --clear-prepared`
3. Обучить модель:
   - `python train_yolov5_windows.py --yolov5-dir yolov5 --data data/person.yaml --weights yolov5s.pt --img 640 --batch 8 --epochs 100 --device 0`
4. Запустить детекцию и подсчет:
   - `python YoloDsortDetection.py --weights "yolov5/runs/train/person_yolov5/weights/best.pt" --video "8.avi" --output "output.avi"`

## Ссылка на датасетик


