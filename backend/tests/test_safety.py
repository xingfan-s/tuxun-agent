import numpy as np
import cv2


def test_face_detect_no_faces():
    from app.safety.face_detect import count_faces
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = count_faces(img)
    assert result == 0


def test_face_detect_cascade_exists():
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    assert not face_cascade.empty()
