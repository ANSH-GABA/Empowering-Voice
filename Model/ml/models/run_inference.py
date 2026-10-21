import os
import sys
import cv2
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from torchvision import transforms
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Model/ml/models
ML_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))# Model/ml
MODEL_ROOT = os.path.abspath(os.path.join(ML_DIR, ".."))# Model/

if MODEL_ROOT not in sys.path:
    sys.path.append(MODEL_ROOT)

from ml.data.download_assets import download_all_assets
from ml.features.model_pipeline import load_pre_trained_classifier

# 1. Pipeline Asset Ingestion
_, model_path, mediapipe_path = download_all_assets()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_pre_trained_classifier(model_path, device)

# 2. Setup MediaPipe Tracking Framework
base_options = python.BaseOptions(model_asset_path=mediapipe_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)
hands = vision.HandLandmarker.create_from_options(options)

# 3. Vision Transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

idx_to_class = {i: chr(65+i) for i in range(26)}  # Matrix tracking map A-Z
cap = cv2.VideoCapture(0)
timestamp_ms = 0

print("\n In-Session Inference active. Press 'q' inside video display window to terminate.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    results = hands.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 33  # ~30 Frames Per Second
    
    # FIX 2: Added [0] here so Python targets the actual point values array of the first hand
    if results.hand_landmarks and len(results.hand_landmarks) > 0:
        landmarks = results.hand_landmarks[0]
        h, w = frame.shape[:2]
        
        # Pull direct floats cleanly out of the MediaPipe point nodes
        x_coords = [lm.x * w for lm in landmarks]
        y_coords = [lm.y * h for lm in landmarks]
        
        # Create a bounding box around the tracked hand boundaries
        x_min, y_min = max(0, int(min(x_coords)) - 40), max(0, int(min(y_coords)) - 40)
        x_max, y_max = min(w, int(max(x_coords)) + 40), min(h, int(max(y_coords)) + 40)
        
        hand_crop = frame[y_min:y_max, x_min:x_max]
        
        if hand_crop.size > 0:
            pil_image = Image.fromarray(cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB))
            
            # FIX: Explicitly cast or type-annotate 'img_tensor' so Pylance knows it's a PyTorch Tensor
            img_tensor: torch.Tensor = preprocess(pil_image)  # type: ignore
            tensor_batch = img_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(tensor_batch)
                probabilities = torch.softmax(outputs, dim=1)
                top_prob, top_idx = torch.max(probabilities, dim=1)
                
                class_index = int(top_idx.item())
                predicted_class = idx_to_class[class_index]
                confidence = top_prob.item() * 100
                
                # Draw box and text onto user window interface
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (46, 204, 113), 2)
                cv2.putText(frame, f"{predicted_class} ({confidence:.1f}%)", 
                            (x_min, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (46, 204, 113), 2)

    cv2.imshow('Empowering Voice - Live ASL Preview', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()