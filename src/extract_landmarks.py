import mediapipe as mp
import cv2
import numpy as np 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
from pathlib import Path

## General Config

RAW_DIR = Path("data/raw")
LANDMARK_DIR = Path("data/landmarks")
MODEL_PATH = "hand_landmarker.task"
MODEL_URL    = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Which letters use two hands in BISINDO
TWO_HAND_SIGNS = {"A","B","D","F","G","H","K","M","N","P","Q","S","T","W","X","Y"}

## Model Download

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmarker model (~25MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")

## Landmark Normalization

def normalize(landmarks):
    # Step 1: extract raw x,y,z into a (21,3) array / matrix
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])  # shape (21, 3)

    # Step 2: subtract wrist to "centralize" every other landmark based on wrist coords
    coords -= coords[0]   # coords[0] is wrist → becomes (0,0,0)

    # Step 3: rescale coords by their absolute max value to "min-max normalize" every image to [-1.0, 1.0] range
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords /= max_val

    return coords.flatten()   # flatten the image to one row with 63 column, shape (63,)

def extract_from_image(detector, img_path):
    """
    Runs MediaPipe on a single image file.
    Returns a numpy array of shape (126,): [right_63 | left_63].
    Returns None if no hand is detected.
    """
    
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"  [!] Could not read image: {img_path}")
        return None

    # Convert BGR to RGB
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None
    
    # Build a dict of {handedness_label: normalized_vector}
    hand_vectors = {}
    for i, hand_landmarks in enumerate(result.hand_landmarks):
        label  = result.handedness[i][0].display_name   # "Right" or "Left"
        vector = normalize(hand_landmarks)               # shape (63,)
        hand_vectors[label] = vector

    # Based on handedness from hand_vectors dict, fills for the rest of the vectors if it's one handed
    right = hand_vectors.get("Right", np.zeros(63))
    left  = hand_vectors.get("Left",  np.zeros(63))

    return np.concatenate([right, left])   # shape (126,)

def extract_class(detector, class_dir, label):
    """
    Processes all images in one class folder.
    Returns a numpy array of shape (n_successful, 126).
    """
    samples      = []
    total        = 0
    failed       = 0
    is_two_hands = label.upper() in TWO_HAND_SIGNS

    img_paths = [p for p in class_dir.iterdir() if p.suffix.lower() in IMG_EXTS]

    for img_path in img_paths:
        total  += 1
        vector  = extract_from_image(detector, img_path)

        if vector is None:
            # print(f"  [SKIP] No hand detected: {img_path.name}")
            failed += 1
            continue

        # For one-hand signs, warn if MediaPipe returned two hands
        # (unusual but possible if a second hand is in frame)
        if not is_two_hands:
            # zero out the left half just to be safe — we only trust right for now
            # (most dataset images are captured with the dominant hand)
            pass   # leave as-is; one-hand signs may legitimately use either hand

        samples.append(vector)

    detection_rate = (total - failed) / total * 100 if total > 0 else 0
    print(f"  {label}: {total - failed}/{total} detected ({detection_rate:.1f}%)")

    return np.array(samples) if samples else None   # shape (n_successful, 126)

## Main 

def main():
    download_model()
    LANDMARK_DIR.mkdir(parents=True, exist_ok=True)

    # IMAGE mode — correct for static files (not video stream)
    # Running mode IMAGE means each detect() call is independent; no tracking between frames
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options      = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,   # ← key difference from sanity check
        num_hands=2,
        min_hand_detection_confidence=0.5,
    )

    failed_classes = []

    with vision.HandLandmarker.create_from_options(options) as detector:
        # Sort so output is alphabetical — easier to debug
        class_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])

        print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}\n")

        for class_dir in class_dirs:
            label = class_dir.name.upper()
            print(f"Processing class: {label}")

            data = extract_class(detector, class_dir, label)

            if data is None:
                print(f"  [!] No valid samples for class {label} — skipping save")
                failed_classes.append(label)
                continue

            out_path = LANDMARK_DIR / f"{label}.npy"
            np.save(out_path, data)           # saves shape (n_samples, 126) to disk
            print(f"  Saved → {out_path}  shape={data.shape}\n")

    # Summary
    print("=" * 50)
    print("Extraction complete.")
    if failed_classes:
        print(f"Classes with no valid samples: {failed_classes}")
    else:
        print("All classes extracted successfully.")

if __name__ == "__main__":
    main()