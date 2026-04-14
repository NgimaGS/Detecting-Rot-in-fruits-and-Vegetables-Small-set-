import cv2
import numpy as np
import onnxruntime as ort

class PantryEngine:
    def _resource_path(self, relative_path):
        import sys
        import os
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.join(os.path.abspath("."), relative_path)

    def __init__(self):
        # Load the ONNX sessions (completely removing PyTorch and Transformers)
        yolo_path = self._resource_path("yolo_pantry.onnx")
        clip_path = self._resource_path("clip_smart.onnx")
        self.yolo_session = ort.InferenceSession(yolo_path)
        self.clip_session = ort.InferenceSession(clip_path)
        
        self.labels = [
            'FreshApple', 'RottenApple', 
            'FreshBanana', 'RottenBanana', 
            'FreshOrange', 'RottenOrange'
        ]
        
        # Valid COCO indices from the original training script
        # 47: Apple, 46: Banana, 49: Orange, 48: Sandwich (treated as Apple)
        # 75: Vase — YOLO often misclassifies rotten/brown apples as vases
        self.valid_indices = {
            47: [0, 1],  # apple
            46: [2, 3],  # banana
            49: [4, 5],  # orange
            48: [0, 1],  # sandwich -> apple (brown rotten apples)
            75: [0, 1],  # vase -> apple (rotten dark apples misclassified)
        }

    def preprocess_clip(self, image_crop):
        # 1. Resize to 224x224
        resized = cv2.resize(image_crop, (224, 224))
        # 2. Convert to float32
        resized = resized.astype(np.float32)
        # 3. Scale pixels to [0, 1]
        resized = resized / 255.0
        # 4. Normalize with specific mean and std
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        normalized = (resized - mean) / std
        # 5. Transpose to [1, 3, 224, 224] shape (B, C, H, W)
        transposed = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(transposed, axis=0)

    def process_image(self, image_path):
        # Read Image
        bgr_image = cv2.imread(image_path)
        if bgr_image is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = rgb_image.shape[:2]
        
        # ---------- 1. YOLOv8 Preprocessing & Inference ----------
        img_resized = cv2.resize(rgb_image, (640, 640))
        img_float = img_resized.astype(np.float32) / 255.0
        img_tensor = np.transpose(img_float, (2, 0, 1))
        img_tensor = np.expand_dims(img_tensor, axis=0)
        
        yolo_input_name = self.yolo_session.get_inputs()[0].name
        yolo_outputs = self.yolo_session.run(None, {yolo_input_name: img_tensor})
        
        # YOLOv8 output is [1, 84, 8400]
        predictions = np.squeeze(yolo_outputs[0]).T  # shape becomes (8400, 84)
        boxes = predictions[:, :4]
        class_probs = predictions[:, 4:]
        
        # Lower threshold to catch rotten produce that YOLO detects with weak confidence
        conf_threshold = 0.08
        scores = np.max(class_probs, axis=1)
        valid_detections = scores > conf_threshold
        
        boxes = boxes[valid_detections]
        scores = scores[valid_detections]
        class_ids = np.argmax(class_probs[valid_detections], axis=1)
        
        # Calculate x, y, h, w for Non-Maximum Suppression
        final_boxes_nms = []
        for i in range(len(boxes)):
            cx, cy, w, h = boxes[i]
            x = cx - (w / 2)
            y = cy - (h / 2)
            final_boxes_nms.append([float(x), float(y), float(w), float(h)])
            
        # Run NMS algorithm
        indices = cv2.dnn.NMSBoxes(final_boxes_nms, scores.tolist(), conf_threshold, nms_threshold=0.45)
        if isinstance(indices, np.ndarray):
            indices = indices.flatten()
        elif hasattr(indices, '__len__'):
            indices = list(indices)
        
        # ---------- 2. Parsing Detections & CLIP Inference ----------
        inventory = {name: 0 for name in self.labels}
        final_detections = []
        
        if len(indices) > 0:
            for idx in indices.flatten():
                yolo_cls = class_ids[idx]
                
                # We only want fruits mapped in our specialist setup
                if yolo_cls in self.valid_indices:
                    cx, cy, bbox_w, bbox_h = boxes[idx]
                    
                    # Convert to corners
                    x1 = cx - bbox_w / 2
                    y1 = cy - bbox_h / 2
                    x2 = cx + bbox_w / 2
                    y2 = cy + bbox_h / 2
                    
                    # Scale coordinates back to original image size
                    x1 = int(max(0, x1 * orig_w / 640))
                    y1 = int(max(0, y1 * orig_h / 640))
                    x2 = int(min(orig_w, x2 * orig_w / 640))
                    y2 = int(min(orig_h, y2 * orig_h / 640))
                    
                    # Add 20px padding (as seen in original notebook)
                    pad = 20
                    c_x1 = max(0, x1 - pad)
                    c_y1 = max(0, y1 - pad)
                    c_x2 = min(orig_w, x2 + pad)
                    c_y2 = min(orig_h, y2 + pad)
                    
                    crop = rgb_image[c_y1:c_y2, c_x1:c_x2]
                    if crop.size == 0:
                        continue
                        
                    # Custom Preprocessing
                    clip_input = self.preprocess_clip(crop)
                    
                    # CLIP Inference
                    clip_input_name = self.clip_session.get_inputs()[0].name
                    clip_outputs = self.clip_session.run(None, {clip_input_name: clip_input})[0]
                    
                    # Softmax and masking based on specialist valid class indices
        	        # Example: if detected Apple, we only look at FreshApple and RottenApple outputs
                    allowed_idx = self.valid_indices[yolo_cls]
                    logits = clip_outputs[0, allowed_idx]
                    
                    # Pure Numpy Softmax computation
                    exp_L = np.exp(logits - np.max(logits))
                    probs = exp_L / np.sum(exp_L)
                    
                    best_local_idx = np.argmax(logits)
                    final_class_idx = allowed_idx[best_local_idx]
                    
                    res_label = self.labels[final_class_idx]
                    conf = probs[best_local_idx]
                    
                    inventory[res_label] += 1
                    final_detections.append({
                        "label": res_label,
                        "confidence": float(conf),
                        "box": [x1, y1, x2, y2] # Original Image Pixel Coordinates
                    })
                    
        return {
            "inventory": dict(inventory),
            "detections": final_detections
        }
