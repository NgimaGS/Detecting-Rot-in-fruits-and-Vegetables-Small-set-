# Fruit & Vegetable Rot Detector

This project implements a computer vision pipeline to detect the freshness and presence of rot in fruits and vegetables.

## Pipeline Architecture
- **Object Detection (YOLOv8)**: First, YOLOv8 is used to detect the location of fruits/vegetables in the image.
- **Freshness Classification (EfficientNet-B2)**: The detected instances are then classified using an EfficientNet-B2 model to assess their freshness.

## Key Features
- Handles standard fresh produce.
- Detects various forms of rot including:
  - Internal rot (dark cores, browning).
  - Surface mold (grey/white fuzzy growth).
  - Standard discoloration.

## Datasets
The model is trained on a combination of datasets to cover a wide spectrum of freshness:
- `jojogo9/freshness_of_fruits_and_veges_256`
- `Densu341/Fresh-rotten-fruit`

## Dependencies
- PyTorch
- Ultralytics (YOLO)
- Hugging Face `datasets`
- Scikit-learn
- Matplotlib
- Pillow

## Repository Structure
- `rot_detection_project.ipynb`: Main Jupyter Notebook containing the training and inference code.
- `efficientnet_rot_detector.pth`: Trained EfficientNet-B2 model weights.
- `yolov8n.pt`: YOLOv8 model weights.
- Sample images are also included for testing.

