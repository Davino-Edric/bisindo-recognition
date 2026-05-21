import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmarker model (~25MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")

def draw_landmarks(frame, hand_landmarks_list):
    """Draw dots manually — Tasks API doesn't have a built-in draw_landmarks helper."""
    h, w = frame.shape[:2]
    for hand_landmarks in hand_landmarks_list:
        for lm in hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

def inspect_frame(result):
    if not result.hand_landmarks:
        print("\n[!] No hands detected — try again with hand clearly visible.")
        return

    print(f"\nHands detected: {len(result.hand_landmarks)}")

    for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
        handedness = result.handedness[hand_idx][0]   # first (best) classification
        print(f"\n--- Hand {hand_idx + 1}: {handedness.display_name} "
              f"(confidence: {handedness.score:.2f}) ---")
        print(f"  Landmark count: {len(hand_landmarks)}")   # expect 21

        for i, lm in enumerate(hand_landmarks):
            print(f"  [{i:>2}] x={lm.x:.4f}  y={lm.y:.4f}  z={lm.z:.4f}")

    # Preview what the flattened feature vector looks like
    flat = []
    for lm in result.hand_landmarks[0]:
        flat.extend([lm.x, lm.y, lm.z])
    print(f"\nFlattened vector (first hand): length={len(flat)}")
    print(f"  First 9 values (landmarks 0-2): {[round(v, 4) for v in flat[:9]]}")

def main():
    download_model()

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not found — try index 1 or 2")

    print("Press SPACE to capture a frame, Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = detector.detect(mp_image)   # IMAGE mode: synchronous, one frame at a time

        if result.hand_landmarks:
            draw_landmarks(frame, result.hand_landmarks)

        cv2.imshow("MediaPipe Sanity Check", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            inspect_frame(result)
            break
        elif key == ord('q'):
            print("Quit.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()