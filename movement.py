import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Colors for indicators
SAFE_COLOR = (0, 255, 0)  # Green
RISK_COLOR = (0, 0, 255)  # Red

# Threshold angles (example values)
KNEE_VALGUS_THRESHOLD = 170  # Valgus collapse risk
KNEE_FLEXION_THRESHOLD = 30   # Low flexion risk

# Start video capture
cap = cv2.VideoCapture("video.mp4")  # Change to 'video.mp4' for file input

def calculate_angle(a, b, c):
    """Calculate the angle between three points (e.g., hip, knee, ankle)."""
    a = np.array(a)  # Hip
    b = np.array(b)  # Knee
    c = np.array(c)  # Ankle

    ab = a - b
    cb = c - b

    dot_product = np.dot(ab, cb)
    magnitude_ab = np.linalg.norm(ab)
    magnitude_cb = np.linalg.norm(cb)
    
    angle = np.arccos(dot_product / (magnitude_ab * magnitude_cb))
    return np.degrees(angle)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for correct orientation
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get landmark coordinates
        hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * frame.shape[1], landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * frame.shape[0])
        knee = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x * frame.shape[1], landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y * frame.shape[0])
        ankle = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x * frame.shape[1], landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * frame.shape[0])

        # Calculate knee angle
        knee_angle = calculate_angle(hip, knee, ankle)

        # Determine color based on safety thresholds
        color = SAFE_COLOR if knee_angle > KNEE_VALGUS_THRESHOLD else RISK_COLOR

        # Draw indicators
        cv2.circle(frame, (int(knee[0]), int(knee[1])), 10, color, -1)
        cv2.putText(frame, f"Knee Angle: {int(knee_angle)}°", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Feedback messages
        if knee_angle <= KNEE_VALGUS_THRESHOLD:
            cv2.putText(frame, "Warning: Knee Collapsing Inward!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, RISK_COLOR, 2)

    # Display frame
    cv2.imshow("Knee Mechanics Analysis", frame)

    # Stop command
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopping analysis...")
        break

cap.release()
cv2.destroyAllWindows()