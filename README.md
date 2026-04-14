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

## Repository Structure

```
final/
│
├── 📓 Notebooks (Training & Experimentation)
│   ├── CLIP_Freshness_Prediction_and_object_detection.ipynb   ← V2 Final Pipeline
│   ├── Clip_Apple.ipynb                                        ← V2 CLIP Experimentation
│   └── rot_detection_project.ipynb                             ← V1 YOLOv8 + EfficientNet
│
├── 🖥️ Application (Packaged Desktop App)
│   └── pantry_app_release/
│       ├── app.py                  ← CustomTkinter GUI (Phase 3)
│       ├── pantry_engine.py        ← ONNX Inference Engine (Phase 2)
│       ├── yolo_pantry.onnx        ← Exported YOLOv8 model
│       ├── clip_smart.onnx         ← Exported CLIP freshness model
│       └── dist/app.app            ← Double-clickable macOS bundle
│
├── 🤖 Models
│   ├── onnx/
│   │   ├── yolo_pantry.onnx        ← Lightweight YOLOv8 (12 MB)
│   │   └── clip_smart.onnx         ← Lightweight CLIP-MLP (335 MB, split)
│   └── pytorch/
│       ├── clip_freshness_smart.pth   ← Trained CLIP-MLP weights (V2 Best)
│       ├── clip_freshness_pantry.pth  ← Trained CLIP-MLP weights (V2 Alt)
│       ├── efficientnet_rot_detector.pth ← V1 EfficientNet weights
│       └── yolov8n.pt                 ← YOLOv8n base weights
│
├── 🖼️ Test Images
│   └── test_images/
│       ├── apple.webp
│       ├── apples.webp
│       ├── rotten-apple.webp
│       ├── rotten_brown.webp
│       └── rotton_apple.webp
│
├── 🛠️ Scripts
│   ├── export_onnx.py              ← Converts PyTorch models → ONNX
│   └── app.py                      ← Standalone GUI entry point
│
└── 📄 Docs
    ├── README.md
    ├── .gitignore
    └── Project Proposal_ Pantry Management.pdf
```

> **Note:** Large model files (`*.pth`, `*.pt`, `*.onnx.data`) are excluded from Git via `.gitignore` due to GitHub's 100MB file size limit.

