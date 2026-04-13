# Pantry Management & Freshness Detection

This project implements a computer vision pipeline to detect the freshness and presence of rot in fruits and vegetables, alongside object counting and bounding box detection.

## Project Progress & Versions

### V2: CLIP Integration & Object Counting (Current)
The final pipeline successfully integrates OpenAI's **CLIP** model combined with robust object detection for a complete pantry management solution. 
- **Object Detection & Counting**: Accurately identifies and counts the number of produce instances in a single shot.
- **Enhanced Freshness Prediction**: Predicts produce state using specifically trained CLIP models (`clip_freshness_pantry.pth` and `clip_freshness_smart.pth`), offering superior rot and freshness classification compared to earlier baseline models.
- **Working Files**: 
  - `CLIP_Freshness_Prediction_and_object_detection.ipynb`: The final, comprehensive implementation that runs the object counting, detection, and state-of-the-art CLIP predictions.
  - `Clip_Apple.ipynb`: Preliminary experimentation using the CLIP architecture.

### V1: YOLOv8 + EfficientNet-B2 (Initial Pipeline)
The baseline pipeline architecture built in the early stages of the project.
- **Object Detection (YOLOv8)**: Used to detect the location of fruits/vegetables in the image.
- **Freshness Classification (EfficientNet-B2)**: Detected instances are evaluated using an EfficientNet-B2 classifier.
- **Working Files**: 
  - `rot_detection_project.ipynb`
  - `efficientnet_rot_detector.pth` 
  - `yolov8n.pt`

## Key Features
- Detects and counts multiple produce instances in a single frame.
- Handles standard fresh produce and successfully partitions counts by freshness (e.g., "# fresh apples", "# rotten apples").
- Detects various forms of rot including:
  - Internal rot (dark cores, browning).
  - Surface mold (grey/white fuzzy growth).
  - Standard discoloration.

## Datasets
The models are trained and evaluated on combination datasets to cover a wide spectrum of freshness:
- `jojogo9/freshness_of_fruits_and_veges_256`
- `Densu341/Fresh-rotten-fruit`

## Dependencies
- PyTorch
- Ultralytics (YOLO)
- Transformers (from Hugging Face)
- Hugging Face `datasets`
- Scikit-learn
- Matplotlib
- Pillow
