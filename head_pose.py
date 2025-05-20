import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# === [MediaPipe setup] ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# === [3D model points] ===
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

# === [MAE calculation] ===
def calculate_mae(values):
    if len(values) == 0:
        return 0
    mean_val = np.mean(values)
    return np.mean(np.abs(np.array(values) - mean_val))

# === [Webcam setup] ===
cap = cv2.VideoCapture(0)

# Buffer to store pitch history (sliding window)
pitch_history = deque(maxlen=30)  # ~1 second if ~30fps
microsleep_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0].landmark

        image_points = np.array([
            (int(face[i].x * w), int(face[i].y * h)) for i in landmark_indices
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

            # === [Draw 3D axes] ===
            nose_point = image_points[0].astype(int)
            axis_length = 100
            axis = np.float32([[axis_length,0,0], [0,axis_length,0], [0,0,axis_length]])
            imgpts, _ = cv2.projectPoints(axis, rotation_vec, translation_vec, get_camera_matrix(w, h), dist_coeffs)
            imgpts = imgpts.astype(int)

            cv2.line(frame, tuple(nose_point), tuple(imgpts[0].ravel()), (0, 0, 255), 3)  # X - red
            cv2.line(frame, tuple(nose_point), tuple(imgpts[1].ravel()), (0, 255, 0), 3)  # Y - green
            cv2.line(frame, tuple(nose_point), tuple(imgpts[2].ravel()), (255, 0, 0), 3)  # Z - blue

            # === [Display angles] ===
            cv2.putText(frame, f"Pitch (up/down): {pitch:.1f}°", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Yaw (left/right): {yaw:.1f}°", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Roll (tilt): {roll:.1f}°", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # === [MAE check] ===
            mae_pitch = calculate_mae(pitch_history)
            cv2.putText(frame, f"MAE (pitch): {mae_pitch:.2f}°", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
            if pitch > 15:
                cv2.putText(frame, "Possible drowsy (head down)", (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if mae_pitch >= 15:
                cv2.putText(frame, "MICRO-SLEEP DETECTED!", (10, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                microsleep_count += 1
                pitch_history.clear()  # reset sau khi cảnh báo

            # === [Counter display] ===
            cv2.putText(frame, f"Microsleep Count: {microsleep_count}", (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)

    cv2.imshow("Head Pose Detection with MAE", frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()