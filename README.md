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

## Getting Started: How to Run

Follow this order to explore or run the project components:

### 1. Research & Training (`/Notebooks`)
If you want to understand the logic or retrain the models:
- Run `CLIP_Freshness_Prediction_and_object_detection.ipynb`. This is the core research file that contains the full object detection and CLIP-based freshness logic.

### 2. Model Preparation (ONNX Export)
The desktop application requires ONNX versions of the models to run efficiently without PyTorch. If these files (`.onnx` and `.onnx.data`) are missing from the `pantry_app_release/` folder:
1. Ensure the PyTorch weights (`yolov8n.pt` and `clip_freshness_smart.pth`) are in the project root.
2. Run the export script: `python export_onnx.py`.
3. Move the newly generated `yolo_pantry.onnx`, `clip_smart.onnx`, and `clip_smart.onnx.data` into the `pantry_app_release/` folder.

### 3. Running the Desktop Application (`/pantry_app_release`)
To launch the actual management interface:
1. Navigate to the `pantry_app_release/` folder.
2. Activate the virtual environment: `source .venv/bin/activate`.
3. Run the app: `python app.py`.
**Note:** Ensure all `.onnx` and `.onnx.data` files are present in this folder as the app loads them on startup.

### 3. Creating/Running the Standalone Executable
If you want to use the app without a Python environment:
- **Build**: Run `pyinstaller app.spec` inside the `pantry_app_release/` folder.
- **Run**: Open the generated `dist/PantryManager.app` or run the `dist/PantryManager` binary.

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
│       ├── app.py                  ← CustomTkinter GUI (Primary Entry Point)
│       ├── pantry_engine.py        ← ONNX Inference Engine
│       ├── app.spec                ← PyInstaller Configuration
│       ├── yolo_pantry.onnx        ← Exported YOLOv8 model
│       ├── clip_smart.onnx         ← Exported CLIP freshness model
│       ├── dist/                   ← **NEW**: Contains standalone executables
│       │   ├── PantryManager       ← Single binary executable
│       │   └── PantryManager.app   ← macOS Application bundle
│       └── .venv/                  ← Local python environment
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

