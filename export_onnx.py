import os
import shutil
import torch
import torch.nn as nn
from ultralytics import YOLO
from transformers import CLIPVisionModelWithProjection

class CLIPFreshnessSpecialist(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.clip = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.clip.parameters():
            param.requires_grad = False
        self.mlp = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, pixel_values):
        # We process the clip without torch.no_grad() here to allow ONNX tracing to construct the graph
        clip_outputs = self.clip(pixel_values=pixel_values)
        image_embeds = clip_outputs.image_embeds
        return self.mlp(image_embeds)

def main():
    print("========================================")
    print("1. Exporting YOLOv8 model to ONNX")
    print("========================================")
    yolo_model = YOLO("yolov8n.pt")
    # This automatically generates yolov8n.onnx
    yolo_export_path = yolo_model.export(format="onnx")
    
    # Rename the output to match requirements
    if yolo_export_path and os.path.exists(yolo_export_path):
        if os.path.exists("yolo_pantry.onnx"):
            os.remove("yolo_pantry.onnx")
        shutil.move(yolo_export_path, "yolo_pantry.onnx")
        print("--> Successfully saved to 'yolo_pantry.onnx'\n")


    print("========================================")
    print("2. Exporting Custom CLIP model to ONNX")
    print("========================================")
    num_classes = 6  # Derived from target_class_names in training notebook
    device = "cpu"   # Generally recommended to trace/export ONNX from CPU
    
    # Initialize model
    print("Initializing CLIPFreshnessSpecialist...")
    clip_model = CLIPFreshnessSpecialist(num_classes).to(device)
    
    # Load weights
    print("Loading weights from 'clip_freshness_smart.pth'...")
    clip_model.load_state_dict(torch.load("clip_freshness_smart.pth", map_location=device))
    clip_model.eval()

    # Create dummy input [1, 3, 224, 224] representing an RGB image crop
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    # Export to ONNX
    onnx_file_path = "clip_smart.onnx"
    print(f"Running torch.onnx.export to '{onnx_file_path}'...")
    torch.onnx.export(
        clip_model,
        dummy_input,
        onnx_file_path,
        export_params=True,
        opset_version=14,                  # Opset 14 provides great support for Transformers
        do_constant_folding=True,
        input_names=['pixel_values'],
        output_names=['logits'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )
    print(f"--> Successfully saved to '{onnx_file_path}'")
    print("\nExport process complete!")

if __name__ == "__main__":
    main()
