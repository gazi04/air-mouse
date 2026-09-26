import cv2 as cv
import mediapipe as mp
import math
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

WRIST = 0
THUMB_TIP, THUMB_DIP, THUMB_PIP, THUMB_MCP = 4, 3, 2, 1
INDEX_TIP, INDEX_DIP, INDEX_PIP, INDEX_MCP = 8, 7, 6, 5
MIDDLE_TIP, MIDDLE_DIP, MIDDLE_PIP, MIDDLE_MCP = 12, 11, 10, 9
RING_TIP, RING_DIP, RING_PIP, RING_MCP = 16, 15, 14, 13
PINKY_TIP, PINKY_DIP, PINKY_PIP, PINKY_MCP = 20, 19, 18, 17


def distance(landmarks, idx_a, idx_b):
    a, b = landmarks[idx_a], landmarks[idx_b]
    return math.hypot(a.x - b.x, a.y - b.y)

def is_pinch(landmarks, threshold=0.2):
    pinch_dist = distance(landmarks, THUMB_TIP, INDEX_TIP)
    hand_size = distance(landmarks, WRIST, MIDDLE_MCP)
    return (pinch_dist / hand_size) < threshold

def is_finger_up(landmarks, tip_idx, pip_idx):
    return landmarks[tip_idx].y < landmarks[pip_idx].y

def is_index_only_up(landmarks):
    return (
        is_finger_up(landmarks, INDEX_TIP, INDEX_PIP)
        and not is_finger_up(landmarks, MIDDLE_TIP, MIDDLE_PIP)
        and not is_finger_up(landmarks, RING_TIP, RING_PIP)
        and not is_finger_up(landmarks, PINKY_TIP, PINKY_PIP)
    )

def is_scroll_pose(landmarks):
    return (
        is_finger_up(landmarks, INDEX_TIP, INDEX_PIP)
        and is_finger_up(landmarks, MIDDLE_TIP, MIDDLE_PIP)
        and not is_finger_up(landmarks, RING_TIP, RING_PIP)
        and not is_finger_up(landmarks, PINKY_TIP, PINKY_PIP)
    )


def draw_landmarks_on_image(rgb_image, detection_result, was_pinching):
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)
    height, width, _ = annotated_image.shape

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = detection_result.handedness[idx][0].category_name

        pinching = is_pinch(hand_landmarks)

        if pinching and not was_pinching.get(idx, False):
            print("Click")
        was_pinching[idx] = pinching

        # if is_index_only_up(hand_landmarks):
        #   print("Index finger up")

        for mark in hand_landmarks:
          x = int(mark.x * width)
          y = int(mark.y * height)
          cv.circle(annotated_image, (x, y), 5, (0, 200, 0), -1)

    return annotated_image, was_pinching

def main() -> None:
    # Create an HandLandmarker object.
    base_options = python.BaseOptions(model_asset_path=f'{Path(__file__).parent}/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                           num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    was_pinching = {}
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        annotated_image, was_pinching = draw_landmarks_on_image(mp_image.numpy_view(), detection_result, was_pinching)
        
        rbg_frame = cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR)
        # Display the resulting frame
        cv.imshow('Hand Tracking', rbg_frame)
        if cv.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
