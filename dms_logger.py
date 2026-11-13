import csv
from datetime import datetime
import time
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import psutil  # Neu für CPU- und RAM-Messung


def calculate_ear(eye_points, landmarks, img_w, img_h):
  p = np.array(
      [
          (
              landmarks[idx].x * img_w,
              landmarks[idx].y * img_h,
              landmarks[idx].z,
          )
          for idx in eye_points
      ],
      dtype=np.float64,
  )
  v1 = np.linalg.norm(p[1] - p[5])
  v2 = np.linalg.norm(p[2] - p[4])
  h = np.linalg.norm(p[0] - p[3])
  if h == 0:
    return 0.0
  return (v1 + v2) / (2.0 * h)


def calculate_mar(mouth_points, landmarks, img_w, img_h):
  p = np.array(
      [
          (
              landmarks[idx].x * img_w,
              landmarks[idx].y * img_h,
              landmarks[idx].z,
          )
          for idx in mouth_points
      ],
      dtype=np.float64,
  )
  v = np.linalg.norm(p[1] - p[0])
  h = np.linalg.norm(p[3] - p[2])
  if h == 0:
    return 0.0
  return v / h


# 1. CSV-Logdatei initialisieren (inkl. Performance-Spalten)
csv_filename = f"dms_perf_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
csv_file = open(csv_filename, mode="w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(
    [
        "Timestamp",
        "Yaw",
        "Pitch",
        "Roll",
        "EAR",
        "MAR",
        "Head_Pose_State",
        "Blink_Status",
        "Mouth_Status",
        "CPU_Percent",
        "RAM_Percent",
    ]
)
print(f"Performance-Aufzeichnung gestartet in: {csv_filename}")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)

EAR_THRESHOLD = 0.22
MAR_YAWN_THRESHOLD = 0.45
alpha = 0.3

smooth_yaw, smooth_pitch, smooth_roll = 0.0, 0.0, 0.0
eyes_closed_start_time = None

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14, 61, 291]

face_3d_model = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)
landmark_indices = [1, 199, 33, 263, 61, 291]

# Prozess-Objekt für gezielte CPU-Messung unseres Skripts
current_process = psutil.Process()

while cap.isOpened():
  success, image = cap.read()
  if not success:
    break

  image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
  image.flags.writeable = False
  results = face_mesh.process(image)
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  img_h, img_w, _ = image.shape
  avg_ear, mar = 0.0, 0.0
  blink_status, mouth_status = "Kein Gesicht", "Normal"
  pose_state = "Kein Gesicht"

  if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
      landmarks = face_landmarks.landmark

      left_ear = calculate_ear(LEFT_EYE, landmarks, img_w, img_h)
      right_ear = calculate_ear(RIGHT_EYE, landmarks, img_w, img_h)
      avg_ear = (left_ear + right_ear) / 2.0

      current_time = time.time()
      if avg_ear < EAR_THRESHOLD:
        if eyes_closed_start_time is None:
          eyes_closed_start_time = current_time
        closed_duration = current_time - eyes_closed_start_time
        blink_status = f"GESCHLOSSEN ({closed_duration:.1f}s)"
        color_eye = (0, 0, 255)
      else:
        eyes_closed_start_time = None
        blink_status = "Auge offen"
        color_eye = (0, 255, 0)

      mar = calculate_mar(MOUTH, landmarks, img_w, img_h)
      if mar > MAR_YAWN_THRESHOLD:
        mouth_status = "Gaehnen"
        color_mouth = (0, 0, 255)
      else:
        mouth_status = "Normal"
        color_mouth = (200, 200, 200)

      face_2d = []
      for idx in landmark_indices:
        lm = landmarks[idx]
        x, y = int(lm.x * img_w), int(lm.y * img_h)
        face_2d.append([x, y])
        if idx == 1:
          nose_2d = (x, y)

      face_2d = np.array(face_2d, dtype=np.float64)
      focal_length = img_w
      cam_matrix = np.array(
          [[focal_length, 0, img_w / 2], [0, focal_length, img_h / 2], [0, 0, 1]],
          dtype=np.float64,
      )
      dist_matrix = np.zeros((4, 1), dtype=np.float64)

      success_pnp, rot_vec, trans_vec = cv2.solvePnP(
          face_3d_model, face_2d, cam_matrix, dist_matrix
      )

      if success_pnp:
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles = cv2.RQDecomp3x3(rmat)[0]

        raw_pitch = angles[0] * (-1)
        raw_yaw = angles[1] * (-1)
        raw_roll = angles[2]

        smooth_yaw = alpha * raw_yaw + (1 - alpha) * smooth_yaw
        smooth_pitch = alpha * raw_pitch + (1 - alpha) * smooth_pitch
        smooth_roll = alpha * raw_roll + (1 - alpha) * smooth_roll

        pose_state = "Geradeaus"
        if smooth_yaw < -15:
          pose_state = "Links"
        elif smooth_yaw > 15:
          pose_state = "Rechts"
        elif smooth_pitch < -12:
          pose_state = "Oben"
        elif smooth_pitch > 12:
          pose_state = "Unten"

  # Hardware-Auslastung messen (CPU für diesen Prozess & RAM-Gesamt)
  cpu_usage = current_process.cpu_percent(interval=None)
  ram_usage = psutil.virtual_memory().percent

  # UI Overlays im Video (inkl. CPU & RAM)
  cv2.putText(
      image,
      f"Pose: {pose_state} | CPU: {cpu_usage}% | RAM: {ram_usage}%",
      (20, 40),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.6,
      (0, 255, 0),
      2,
  )
  cv2.putText(
      image,
      f"Augen: {blink_status} | Mund: {mouth_status}",
      (20, 80),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.6,
      (255, 255, 0),
      2,
  )

  # In CSV schreiben
  timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
  csv_writer.writerow([
      timestamp_str,
      round(smooth_yaw, 2),
      round(smooth_pitch, 2),
      round(smooth_roll, 2),
      round(avg_ear, 2),
      round(mar, 2),
      pose_state,
      blink_status,
      mouth_status,
      cpu_usage,
      ram_usage,
  ])

  cv2.imshow("DMS Performance Telemetry", image)
  if cv2.waitKey(5) & 0xFF == 27:
    break

cap.release()
cv2.destroyAllWindows()
csv_file.close()

# --- AUSWERTUNG ---
print("\n--- PERFORMANCE- & SITZUNGS-AUSWERTUNG ---")
try:
  df = pd.read_csv(csv_filename)
  print(f"Durchschnittliche CPU-Last (Skript): {df['CPU_Percent'].mean():.1f}%")
  print(f"Durchschnittliche RAM-Last (System): {df['RAM_Percent'].mean():.1f}%")
except Exception as e:
  print(f"Fehler bei der Auswertung: {e}")