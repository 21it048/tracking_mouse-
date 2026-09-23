import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

# PyAutoGUI settings
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
screen_w, screen_h = pyautogui.size()

# MediaPipe Face Mesh Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, # Iris tracking support
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cam = cv2.VideoCapture(0)

prev_x, prev_y = 0, 0
smooth_factor = 0.2
last_click_time = 0

# Eye Aspect Ratio (EAR) - Blink detect panna
def get_ear(landmarks, top_idx, bottom_idx, left_idx, right_idx, img_w, img_h):
    top = np.array([landmarks[top_idx].x * img_w, landmarks[top_idx].y * img_h])
    bottom = np.array([landmarks[bottom_idx].x * img_w, landmarks[bottom_idx].y * img_h])
    left = np.array([landmarks[left_idx].x * img_w, landmarks[left_idx].y * img_h])
    right = np.array([landmarks[right_idx].x * img_w, landmarks[right_idx].y * img_h])

    vert_dist = np.linalg.norm(top - bottom)
    horiz_dist = np.linalg.norm(left - right)

    return vert_dist / horiz_dist

print("Starting MediaPipe Eye Tracker... Press ESC to exit.")

while cam.isOpened():
    success, frame = cam.read()
    if not success:
        print("Camera error!")
        break

    frame = cv2.flip(frame, 1) # Mirror image
    frame_h, frame_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Left Eye Iris Center Landmark (Index 468)
        iris = landmarks[468]
        iris_x = int(iris.x * frame_w)
        iris_y = int(iris.y * frame_h)

        # Kannu pupil mela matum Pachai Dot podrom
        cv2.circle(frame, (iris_x, iris_y), 4, (0, 255, 0), -1)

        # Eye movement range-a screen coordinates-ku map panrom
        # Sensitivity-a adjust panna 0.35 / 0.65 values-a tweak pannalam
        norm_x = (iris.x - 0.35) / (0.65 - 0.35)
        norm_y = (iris.y - 0.35) / (0.65 - 0.35)

        norm_x = max(0, min(1, norm_x))
        norm_y = max(0, min(1, norm_y))

        target_x = screen_w * norm_x
        target_y = screen_h * norm_y

        # Smooth Cursor Movement
        curr_x = prev_x + (target_x - prev_x) * smooth_factor
        curr_y = prev_y + (target_y - prev_y) * smooth_factor

        pyautogui.moveTo(curr_x, curr_y)
        prev_x, prev_y = curr_x, curr_y

        # Blink Detection (Right Eye Index)
        right_ear = get_ear(landmarks, 386, 374, 362, 263, frame_w, frame_h)

        current_time = time.time()
        if right_ear < 0.18 and (current_time - last_click_time > 0.5):
            pyautogui.click()
            last_click_time = current_time
            cv2.putText(frame, "BLINK CLICK!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Eye Tracking Mouse', frame)

    if cv2.waitKey(1) & 0xFF == 27: # Press ESC to stop
        break

cam.release()
cv2.destroyAllWindows()