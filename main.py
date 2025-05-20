import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import time
from collections import deque
from playsound import playsound
import threading
import pygame
import os
pygame.mixer.init()

# Load models
model_yawn = load_model('models/yawn_detection_model.h5')
model_eye = load_model('models/eye_closures_model.h5')

# Alert sound paths
sound_dir = os.path.join(os.path.dirname(__file__), 'alert_sounds')
SOUND_PERCLOSE = os.path.join(sound_dir, 'perclose_warning.mp3')
SOUND_CONTINUOUS_EYE = os.path.join(sound_dir, 'continous_closed_eye.mp3')
SOUND_YAWN = os.path.join(sound_dir, 'yawn_alert.mp3')
SOUND_HEAD_DOWN = os.path.join(sound_dir, 'head_down.mp3')

# Load sound files safely
sounds_loaded = True
try:
    perclose_sound = pygame.mixer.Sound(SOUND_PERCLOSE)
except Exception as e:
    print(f"Failed to load SOUND_PERCLOSE: {e}")
    sounds_loaded = False

try:
    continuous_eye_sound = pygame.mixer.Sound(SOUND_CONTINUOUS_EYE)
except Exception as e:
    print(f"Failed to load SOUND_CONTINUOUS_EYE: {e}")
    sounds_loaded = False

try:
    yawn_sound = pygame.mixer.Sound(SOUND_YAWN)
except Exception as e:
    print(f"Failed to load SOUND_YAWN: {e}")
    sounds_loaded = False

try:
    head_down_sound = pygame.mixer.Sound(SOUND_HEAD_DOWN)
except Exception as e:
    print(f"Failed to load SOUND_HEAD_DOWN: {e}")
    sounds_loaded = False
    
def play_sound_sync(sound):
    """Plays a sound and waits for it to finish. Prevents overlap."""
    if pygame.mixer.get_busy():
        pygame.mixer.stop()  # Stop any currently playing sound
    channel = sound.play()
    while channel.get_busy():
        pygame.time.wait(100)  # Wait in small increments until the sound ends


# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Constants
img_size = (224, 224)
cap = cv2.VideoCapture(0)

# Landmark indices
LEFT_EYE_IDX = [33, 133, 159, 145]
RIGHT_EYE_IDX = [362, 263, 386, 374]
MOUTH_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]

# 3D model points for head pose
model_points = np.array([
    (0.0, 0.0, 0.0),           # Nose tip
    (0.0, -63.6, -12.5),       # Chin
    (-43.3, 32.7, -26.0),      # Left eye corner
    (43.3, 32.7, -26.0),       # Right eye corner
    (-28.9, -28.9, -24.1),     # Left mouth corner
    (28.9, -28.9, -24.1)       # Right mouth corner
], dtype=np.float32)

landmark_indices = [1, 152, 33, 263, 61, 291]

def get_camera_matrix(w, h):
    focal_length = w
    return np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1]
    ], dtype=np.float32)

dist_coeffs = np.zeros((4, 1))  # no lens distortion

# Detection counters and timers
eye_closed_frames = 0
total_frames = 0
yawn_count = 0
start_time = time.time()
perclos_window_start = time.time()
yawn_window_start = time.time()

# Eye closure duration tracking
eye_closed_start_time = None
min_eye_closed_duration = 3  # seconds

# Head pose variables
pitch_history = deque(maxlen=30) 
head_down_warning = False

# Alert flags
eye_alert = False
yawn_alert = False
head_alert = False
continuous_eye_alert = False  # For continuous eye closure detection

# Yawn detection variables
yawn_start_time = None
min_yawn_duration = 6  # seconds
is_continuous_yawn = False
current_yawn_duration = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    # Default statuses
    is_yawning = False
    left_eye_closed = False
    right_eye_closed = False
    pitch = 0
    yaw = 0
    roll = 0

    if result.multi_face_landmarks:
        face_landmarks = result.multi_face_landmarks[0]

        # ---------- YAWN DETECTION ----------
        mouth_pts = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in MOUTH_IDX]
        mx1 = max(min(p[0] for p in mouth_pts) - 20, 0)
        my1 = max(min(p[1] for p in mouth_pts) - 20, 0)
        mx2 = min(max(p[0] for p in mouth_pts), w)
        my2 = min(max(p[1] for p in mouth_pts), h)

        mouth_crop = frame[my1:my2, mx1:mx2]
        yawn_status = "Mouth Not Found"

        if mouth_crop.size > 0:
            mouth_input = cv2.resize(mouth_crop, img_size).astype('float32') / 255.0
            mouth_input = np.expand_dims(mouth_input, axis=0)
            yawn_pred = model_yawn.predict(mouth_input)

            if yawn_pred[0] > 0.5:
                yawn_status = 'YAWN'    
                is_yawning = True
            else:
                yawn_status = 'Not Yawn'

            cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0, 0, 255), 2)
            cv2.putText(frame, yawn_status, (mx1, my1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ---------- EYE DETECTION ----------
        def get_eye_box(indices):
            pts = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in indices]
            x_vals = [pt[0] for pt in pts]
            y_vals = [pt[1] for pt in pts]
            return min(x_vals), min(y_vals), max(x_vals), max(y_vals)

        for label, idx in zip(['Left Eye', 'Right Eye'], [LEFT_EYE_IDX, RIGHT_EYE_IDX]):
            ex1, ey1, ex2, ey2 = get_eye_box(idx)
            eye_crop = frame[ey1:ey2, ex1:ex2]

            if eye_crop.size > 0:
                eye_input = cv2.resize(eye_crop, img_size).astype('float32') / 255.0
                eye_input = np.expand_dims(eye_input, axis=0)
                eye_pred = model_eye.predict(eye_input)

                corrected = 1 - eye_pred[0]  # Flip the prediction: 0.0 = open, 1.0 = closed
                closed = corrected > 0.7

                if label == 'Left Eye':
                    left_eye_closed = closed
                else:
                    right_eye_closed = closed

                eye_status = f'{label}: {"Closed" if closed else "Open"}'
                cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (255, 0, 0), 2)
                cv2.putText(frame, eye_status, (ex1, ey1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # ---------- HEAD POSE DETECTION ----------
        
        image_points = np.array([
            (int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in landmark_indices
        ], dtype=np.float32)

        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points,
            image_points,
            get_camera_matrix(w, h),
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            rot_matrix, _ = cv2.Rodrigues(rotation_vec)
            proj_matrix = np.hstack((rot_matrix, translation_vec))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
            pitch, yaw, roll = [angle[0] for angle in euler_angles]

            # Normalize pitch
            if pitch > 90: pitch -= 180
            elif pitch < -90: pitch += 180

            # Append pitch to history
            pitch_history.append(pitch)

            # Draw 3D axes
            nose_point = image_points[0].astype(int)
            axis_length = 100
            axis = np.float32([[axis_length,0,0], [0,axis_length,0], [0,0,axis_length]])
            imgpts, _ = cv2.projectPoints(axis, rotation_vec, translation_vec, get_camera_matrix(w, h), dist_coeffs)
            imgpts = imgpts.astype(int)

            cv2.line(frame, tuple(nose_point), tuple(imgpts[0].ravel()), (0, 0, 255), 3)  # X - red
            cv2.line(frame, tuple(nose_point), tuple(imgpts[1].ravel()), (0, 255, 0), 3)  # Y - green
            cv2.line(frame, tuple(nose_point), tuple(imgpts[2].ravel()), (255, 0, 0), 3)  # Z - blue

    # ---------- PERCLOS CALCULATION (1-minute window) ----------
    '''
    - PERCLOS: number of frame detected as closed eye / total number of frame in 1 minute >=0.7 => warning
    - Eye detected as closed continuously for more than 3s => warning

    '''
    
    total_frames += 1
    both_eyes_closed = left_eye_closed and right_eye_closed
    
    if both_eyes_closed:
        eye_closed_frames += 1
    
    current_time = time.time()
    if current_time - perclos_window_start >= 60:  # 60 seconds = 1 minute
        perclos = eye_closed_frames / total_frames
        if perclos >= 0.7:
            eye_alert = True
        else:
            eye_alert = False
        
        # Reset counters for next window
        eye_closed_frames = 0
        total_frames = 0
        perclos_window_start = current_time
    
    # Check for continuous eye closure: if eye is closed for more than 3 seconds=> warning
    if both_eyes_closed:
        if eye_closed_start_time is None:
            eye_closed_start_time = current_time
        elif current_time - eye_closed_start_time >= min_eye_closed_duration:
            continuous_eye_alert = True
    else:
        eye_closed_start_time = None
        continuous_eye_alert = False
        
    # ---------- YAWN DETECTION (15-minute window) ----------
    
    # Yawn duration tracking
    if is_yawning:
        if yawn_start_time is None:
            yawn_start_time = current_time
            current_yawn_duration = 0
        else:
            current_yawn_duration = current_time - yawn_start_time
        
        # Only count the yawn if it lasts at least min_yawn_duration seconds
        if current_yawn_duration >= min_yawn_duration and not is_continuous_yawn:
            yawn_count += 1
            is_continuous_yawn = True
    else:
        # Reset yawn tracking if mouth is not yawning
        yawn_start_time = None
        is_continuous_yawn = False
        current_yawn_duration = 0
    
    # Check if in 15 minutes, they have yawned 3 times
    if current_time - yawn_window_start <= 900:  # 15 minutes
        if yawn_count >= 3:
            yawn_alert = True
            yawn_count = 0  # Reset yawn count after alert
            yawn_window_start = current_time  # Reset yawn window start time
        else:
            yawn_alert = False

    else: # if 15 minutes have passed
        yawn_window_start = current_time  # Reset yawn window start time
        yawn_count = 0  # Reset yawn count after 15 minutes
        yawn_alert = False        

    # ---------- HEAD POSE DETECTION ----------
    if len(pitch_history) > 0:
        current_pitch = pitch_history[-1]
        if current_pitch > 20:  # Head down threshold
            head_alert = True
        else:
            head_alert = False

    # ---------- DISPLAY ALERTS ----------
    alert_y_position = 30
    if eye_alert:
        cv2.putText(frame, "ALERT: PERCLOS >= 70% (Drowsy Eyes)", (10, alert_y_position),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        alert_y_position += 30
        # Play sound for PERCLOS alert
        play_sound_sync(perclose_sound)
    
    if continuous_eye_alert:
        cv2.putText(frame, "ALERT: Eyes Closed > 3s", (10, alert_y_position),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        alert_y_position += 30
        # Play sound for continuous eye closure alert
        play_sound_sync(continuous_eye_sound)
    
    if yawn_alert:
        cv2.putText(frame, "ALERT: Excessive Yawning (>=3 in 15min)", (10, alert_y_position),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        alert_y_position += 30
        # Play sound for yawn alert
        play_sound_sync(yawn_sound)
    
    if head_alert:
        cv2.putText(frame, "ALERT: Head Down Detected", (10, alert_y_position),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        alert_y_position += 30
        # Play sound for head down alert
        play_sound_sync(head_down_sound)
        print("Head down alert sound played")
        

    # ---------- DISPLAY STATISTICS ----------
    cv2.putText(frame, f'Eye Closed Frames: {eye_closed_frames}/{total_frames}', (10, h - 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
    cv2.putText(frame, f'Yawn Count: {yawn_count}', (10, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
    if len(pitch_history) > 0:
        cv2.putText(frame, f'Head Pitch: {pitch_history[-1]:.1f}°', (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

    # Display yawn duration if currently yawning
    if is_yawning and yawn_start_time is not None:
        cv2.putText(frame, f'Yawn Duration: {current_yawn_duration:.1f}s', (10, h - 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

    # Display time remaining in current yawn window
    time_remaining = max(0, 900 - (current_time - yawn_window_start))
    # cv2.putText(frame, f'Yawn Window: {time_remaining:.1f}s remaining', (10, h - 150),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

    # ---------- SHOW FRAME ----------
    cv2.imshow("Drowsiness Detection System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()