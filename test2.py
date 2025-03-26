import cv2
import torch
import clip
import numpy as np
from PIL import Image
import os

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model, preprocess = clip.load("ViT-B/32", device=device)
    print("CLIP model loaded successfully!")
except Exception as e:
    print(f"Error loading CLIP model: {e}")
    exit()

# Initialize video capture (webcam or video file)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # For webcam
# cap = cv2.VideoCapture("video.mp4")  # For video file

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Initialize Selective Search
try:
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
except Exception as e:
    print(f"Error initializing Selective Search: {e}")
    exit()

# Generic object categories
generic_objects = [
    "a person", "a pet", "a vehicle", "a plant", "a screen", "a book", 
    "a bottle", "a food item", "a piece of furniture", "an electronic device"
]

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Selective Search
    ss.setBaseImage(frame)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()

    detected_objects = []
    max_proposals = min(len(rects), 20)  # Limit proposals for speed

    for (x, y, w, h) in rects[:max_proposals]:
        if w < 50 or h < 50:  # Skip small regions
            continue

        roi = frame[y:y+h, x:x+w]
        roi = cv2.resize(roi, (224, 224))  # CLIP expects 224x224

        # Preprocess for CLIP
        pil_image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        image_input = preprocess(pil_image).unsqueeze(0).to(device)

        # CLIP inference
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_inputs = torch.cat([clip.tokenize(obj) for obj in generic_objects]).to(device)
            text_features = model.encode_text(text_inputs)
            similarity = (image_features @ text_features.T).softmax(dim=-1)
            scores = similarity.cpu().numpy()[0]

        max_idx = np.argmax(scores)
        confidence = scores[max_idx]
        if confidence > 0.85:  # High-confidence detections only
            detected_objects.append((x, y, w, h, generic_objects[max_idx], confidence))

    # Draw bounding boxes
    for (x, y, w, h, obj, conf) in detected_objects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{obj} ({conf:.2f})", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Dynamic Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()