import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

LEFT_IRIS_CENTER = 468   
RIGHT_IRIS_CENTER = 473  

SCREEN_W, SCREEN_H = pyautogui.size()
EAR_THRESHOLD = 0.24        
BLINK_CONSEC_FRAMES = 3       
SMOOTHING = 0.25              

pyautogui.FAILSAFE = True     
pyautogui.PAUSE = 0           

def eye_aspect_ratio(landmarks, eye_indices):
    p1 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
    p2 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
    p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
    p4 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
    p5 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
    p6 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

cv2.namedWindow('Eye Controlled Mouse', cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow('Eye Controlled Mouse', 800, 600)

cam = cv2.VideoCapture(0)
cam_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
smooth_x, smooth_y = SCREEN_W // 2, SCREEN_H // 2

left_blink_counter = 0
right_blink_counter = 0

print(f"Screen: {SCREEN_W}x{SCREEN_H} | Cam: {cam_w}x{cam_h}")
print("Controls:")
print("  • Move your eyes to move the cursor")
print("  • Blink LEFT  eye → LEFT  click")
print("  • Blink RIGHT eye → RIGHT click")
print("  • Press ESC to quit")

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0] 

            left_iris = landmarks[LEFT_IRIS_CENTER]
            right_iris = landmarks[RIGHT_IRIS_CENTER]

            iris_x = (left_iris.x + right_iris.x) / 2.0
            iris_y = (left_iris.y + right_iris.y) / 2.0

            margin_x, margin_y = 0.15, 0.20
            target_x = np.interp(iris_x, [margin_x, 1.0 - margin_x], [SCREEN_W, 0])
            target_y = np.interp(iris_y, [margin_y, 1.0 - margin_y], [0, SCREEN_H])
            
            target_x = max(1, min(SCREEN_W - 2, target_x))
            target_y = max(1, min(SCREEN_H - 2, target_y))

            smooth_x += (target_x - smooth_x) * SMOOTHING
            smooth_y += (target_y - smooth_y) * SMOOTHING

            final_x = int(max(1, min(SCREEN_W - 2, smooth_x)))
            final_y = int(max(1, min(SCREEN_H - 2, smooth_y)))
            
            pyautogui.moveTo(final_x, final_y)

            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)

            if left_ear < EAR_THRESHOLD and right_ear >= EAR_THRESHOLD:
                left_blink_counter += 1
            else:
                if left_blink_counter >= BLINK_CONSEC_FRAMES:
                    pyautogui.click(button='left')
                    print("🖱️  LEFT click", flush=True)
                left_blink_counter = 0

            if right_ear < EAR_THRESHOLD and left_ear >= EAR_THRESHOLD:
                right_blink_counter += 1
            else:
                if right_blink_counter >= BLINK_CONSEC_FRAMES:
                    pyautogui.click(button='right')
                    print("🖱️  RIGHT click", flush=True)
                right_blink_counter = 0

            cv2.circle(frame, (int(left_iris.x * w), int(left_iris.y * h)), 3, (0, 255, 0), -1)
            cv2.circle(frame, (int(right_iris.x * w), int(right_iris.y * h)), 3, (0, 255, 0), -1)

            for eye_indices, color in [(LEFT_EYE, (255, 200, 0)), (RIGHT_EYE, (0, 200, 255))]:
                pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices], np.int32)
                cv2.polylines(frame, [pts], True, color, 1)
        else:
            left_ear, right_ear = 1.0, 1.0

        mirrored_frame = cv2.flip(frame, 1)

        draw_x = int(np.interp(smooth_x, [0, SCREEN_W], [0, w]))
        draw_y = int(np.interp(smooth_y, [0, SCREEN_H], [0, h]))
        
        cv2.circle(mirrored_frame, (draw_x, draw_y), 8, (0, 0, 255), -1)
        cv2.circle(mirrored_frame, (draw_x, draw_y), 10, (255, 255, 255), 2)

        if result.face_landmarks:
            cv2.putText(mirrored_frame, f"L-EAR: {left_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            cv2.putText(mirrored_frame, f"R-EAR: {right_ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        else:
            cv2.putText(mirrored_frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Eye Controlled Mouse', mirrored_frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

cam.release()
cv2.destroyAllWindows()