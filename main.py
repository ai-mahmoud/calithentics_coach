import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import time
from pose_utils import get_landmark_raw, calculate_angle

# ─────────────────────────────────────────────
#  MediaPipe Tasks API Setup  (works on 0.10.x)
# ─────────────────────────────────────────────
#
#  The old "solutions" API is gone in 0.10.x.
#  The new "Tasks API" works the same way under the hood —
#  we just initialize differently and call detect_for_video()
#  instead of process() each frame.

MODEL_PATH = "pose_landmarker_lite.task"

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.7,
    min_pose_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.PoseLandmarker.create_from_options(options)


# ─────────────────────────────────────────────
#  Landmark Index Reference (same as before)
# ─────────────────────────────────────────────
# 11 = Left Shoulder    12 = Right Shoulder
# 13 = Left Elbow       14 = Right Elbow
# 15 = Left Wrist       16 = Right Wrist
# 23 = Left Hip         24 = Right Hip
# 25 = Left Knee        26 = Right Knee
# 27 = Left Ankle       28 = Right Ankle

SKELETON_CONNECTIONS = [
    (11, 12),
    (11, 23), (12, 24),
    (23, 24),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]


def landmarks_to_pixels(landmarks, image_shape):
    """Convert normalized landmark list → pixel (x, y) dict keyed by index."""
    h, w = image_shape[:2]
    return {
        i: (int(lm.x * w), int(lm.y * h))
        for i, lm in enumerate(landmarks)
    }


def draw_skeleton(frame, pixel_landmarks):
    """Draw bones (lines) and joints (circles) on the frame."""
    for start_idx, end_idx in SKELETON_CONNECTIONS:
        if start_idx in pixel_landmarks and end_idx in pixel_landmarks:
            cv2.line(frame,
                     pixel_landmarks[start_idx],
                     pixel_landmarks[end_idx],
                     (0, 200, 255), 2, cv2.LINE_AA)
    for idx, point in pixel_landmarks.items():
        cv2.circle(frame, point, 5, (255, 255, 255), -1)
        cv2.circle(frame, point, 7, (0, 200, 255), 1)


def draw_angle_on_joint(frame, point, angle):
    cv2.putText(frame, f"{int(angle)}",
                (point[0] + 10, point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 0), 2, cv2.LINE_AA)


def draw_hud(frame, exercise_name, angle, status):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, exercise_name,
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Angle: {int(angle)} deg",
                (15, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 1, cv2.LINE_AA)

    color = (0, 255, 0) if status in ("UP", "DOWN") else (0, 100, 255)
    cv2.putText(frame, status,
                (w - 180, 45), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, color, 2, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("=" * 40)
    print("  Calisthenics AI Coach — RUNNING")
    print("  Press Q to quit")
    print("=" * 40)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(time.time() * 1000)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            pixel_lm = landmarks_to_pixels(landmarks, frame.shape)

            draw_skeleton(frame, pixel_lm)

            shoulder = pixel_lm.get(11)
            elbow    = pixel_lm.get(13)
            wrist    = pixel_lm.get(15)

            if shoulder and elbow and wrist:
                angle = calculate_angle(shoulder, elbow, wrist)

                for pt in [shoulder, elbow, wrist]:
                    cv2.circle(frame, pt, 9, (0, 255, 255), -1)
                    cv2.circle(frame, pt, 11, (255, 255, 255), 2)

                draw_angle_on_joint(frame, elbow, angle)

                if angle > 150:
                    status = "UP"
                elif angle < 90:
                    status = "DOWN"
                else:
                    status = "MOVING"

                draw_hud(frame, "PUSH-UP [placeholder]", angle, status)

        else:
            cv2.putText(frame, "No pose detected — step into frame",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 100, 255), 2, cv2.LINE_AA)

        cv2.imshow("Calisthenics AI Coach", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
