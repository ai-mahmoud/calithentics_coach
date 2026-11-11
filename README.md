# Calisthenics Coach (prototype)

Real-time exercise feedback from a webcam. MediaPipe Pose tracks the body, the app measures joint angles, and it tells you where you are in each repetition.

**Status:** early prototype. The push-up check (elbow angle: above 150° is up, below 90° is down) works end to end, and the other exercises are still to come.

## What it does

- Detects 33 body landmarks per frame with the MediaPipe Tasks API (`pose_landmarker_lite.task`, video mode).
- Draws the skeleton and shows joint angles live on screen.
- Shows the exercise name, the current angle and the movement status in an on-screen panel.
- `kivy/` contains a Kivy version of the camera app, the first step toward a mobile build.

## Run it

```sh
pip install opencv-python mediapipe
python main.py
```

**Built with:** Python, OpenCV, MediaPipe, Kivy
