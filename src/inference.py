from model import BisindoMLP
from extract_landmarks import MODEL_PATH, normalize, download_model

import mediapipe as mp 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import json
import numpy as np
import torch
import cv2

download_model()
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)

def load_model(model_path,label_map_path):
    model = BisindoMLP()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    return model, label_map

def predict(frame, model, label_map):
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None, None

    hand_vectors = {}
    for i, hand_landmarks in enumerate(result.hand_landmarks):
        label  = result.handedness[i][0].display_name
        hand_vectors[label] = normalize(hand_landmarks)

    right  = hand_vectors.get("Right", np.zeros(63))
    left   = hand_vectors.get("Left",  np.zeros(63))
    vector = np.concatenate([right, left])

    with torch.no_grad():
        tensor         = torch.tensor(vector, dtype=torch.float32).unsqueeze(0)
        output         = model(tensor)
        predicted_idx  = torch.argmax(output, dim=1).item()
        predicted_letter = label_map.get(str(predicted_idx), "?")

    return predicted_letter, vector

if __name__ == "__main__":
    model, label_map = load_model("../best_model.pth", "../label_map.json")
    
    frame = cv2.imread("../data/raw/A/augmented_image_0.jpg")
    letter, _ = predict(frame, model, label_map)
    print(f"Predicted letter: {letter}")