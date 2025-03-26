import cv2
import torch
import clip
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Initialize webcam
cap = cv2.VideoCapture("video.mp4", cv2.CAP_DSHOW)  # Force FFmpegv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Selective Search for Region Proposals
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# Predefined generic categories (optional, can be replaced with AI-generated text)
generic_objects = [
    "a person", "a pet", "a vehicle", "a plant", "a screen", "a book", "a bottle", "a food item",
    "a piece of furniture", "an electronic device", "a cloth", "a bag", "a ball"
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply Selective Search
    ss.setBaseImage(frame)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()

    detected_objects = []
    max_proposals = min(len(rects), 20)  # Reduce proposals for efficiency

    for (x, y, w, h) in rects[:max_proposals]:
        if w < 50 or h < 50:  # Ignore small objects
            continue

        roi = frame[y:y+h, x:x+w]
        roi = cv2.resize(roi, (224, 224))  # Resize for CLIP

        # Convert to PIL and preprocess
        pil_image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        image_input = preprocess(pil_image).unsqueeze(0).to(device)

        # Convert image to text dynamically
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_inputs = torch.cat([clip.tokenize(obj) for obj in generic_objects]).to(device)
            text_features = model.encode_text(text_inputs)

        # Compute similarity
        similarity_scores = (image_features @ text_features.T).softmax(dim=-1)
        scores = similarity_scores.cpu().numpy()[0]

        max_score_index = np.argmax(scores)
        detected_object = generic_objects[max_score_index]
        confidence = scores[max_score_index]

        if confidence > 0.85:  # Draw bounding box only for confident detections
            detected_objects.append((x, y, w, h, detected_object, confidence))

    # Draw bounding boxes
    for (x, y, w, h, obj, conf) in detected_objects:
        label = f"{obj} ({conf:.2f})"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display frame
    cv2.imshow("Dynamic Multi-Object Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
