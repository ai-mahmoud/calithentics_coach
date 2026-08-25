import numpy as np


def calculate_angle(a, b, c):
    """
    Calculate the angle at joint B, formed by points A → B → C.

    Think of it like this:
        A = one end of the limb  (e.g. shoulder)
        B = the joint we care about (e.g. elbow)
        C = other end of the limb  (e.g. wrist)

    We turn the two line segments (BA and BC) into vectors,
    then use the dot product formula to find the angle between them.

    Returns angle in degrees (0° = fully bent, 180° = fully straight).
    """
    a = np.array(a)  # e.g. shoulder
    b = np.array(b)  # e.g. elbow  ← angle measured here
    c = np.array(c)  # e.g. wrist

    # Vectors pointing AWAY from the joint
    ba = a - b
    bc = c - b

    # cos(θ) = (BA · BC) / (|BA| * |BC|)
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)

    # Clamp to [-1, 1] to avoid floating point errors in arccos
    cosine = np.clip(cosine, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine))
    return angle


def get_landmark(landmarks, index, image_shape):
    """
    Extract a single landmark as (x, y) pixel coordinates.

    MediaPipe gives landmarks as normalized floats [0.0 → 1.0].
    We multiply by the image dimensions to get actual pixel positions.
    """
    h, w = image_shape[:2]
    lm = landmarks[index]
    return (int(lm.x * w), int(lm.y * h))


def get_landmark_raw(landmarks, index):
    """
    Extract a landmark as a raw [x, y] array (still normalized).
    Used when we only need proportional positions, not pixel coords.
    (e.g. checking if knee is over toes — relative position matters, not pixels)
    """
    lm = landmarks[index]
    return [lm.x, lm.y]
