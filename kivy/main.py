import cv2
import time
import numpy as np
import mediapipe as mp

from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from pose_utils import calculate_angle

# ─────────────────────────────────────────────
# MediaPipe Setup
# ─────────────────────────────────────────────
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

# Skeleton connections
SKELETON_CONNECTIONS = [
    (11, 12),
    (11, 23), (12, 24),
    (23, 24),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────
def landmarks_to_pixels(landmarks, shape):
    h, w = shape[:2]
    return {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(landmarks)}


def draw_skeleton(frame, lm):
    for a, b in SKELETON_CONNECTIONS:
        if a in lm and b in lm:
            cv2.line(frame, lm[a], lm[b], (0, 200, 255), 2)
    for p in lm.values():
        cv2.circle(frame, p, 5, (255, 255, 255), -1)


def draw_hud(frame, angle, status):
    cv2.putText(frame, f"Angle: {int(angle)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    cv2.putText(frame, status,
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2)


# ─────────────────────────────────────────────
# Kivy App
# ─────────────────────────────────────────────
class CamApp(App):
    def build(self):
        self.img = Image()
        self.cap = cv2.VideoCapture(0)

        Clock.schedule_interval(self.update, 1.0 / 30.0)
        return self.img

    def update(self, dt):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)

        # MediaPipe processing
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(time.time() * 1000)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            pixel_lm = landmarks_to_pixels(landmarks, frame.shape)

            draw_skeleton(frame, pixel_lm)

            s = pixel_lm.get(11)
            e = pixel_lm.get(13)
            w = pixel_lm.get(15)

            if s and e and w:
                angle = calculate_angle(s, e, w)

                if angle > 150:
                    status = "UP"
                elif angle < 90:
                    status = "DOWN"
                else:
                    status = "MOVING"

                draw_hud(frame, angle, status)

        # Convert to texture for Kivy
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

        self.img.texture = texture

    def on_stop(self):
        self.cap.release()
        detector.close()


if __name__ == "__main__":
    CamApp().run()
