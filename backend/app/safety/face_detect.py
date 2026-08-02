from pathlib import Path

import cv2
import numpy as np

_cascade: cv2.CascadeClassifier | None = None


def _get_cascade() -> cv2.CascadeClassifier:
    global _cascade
    if _cascade is None:
        _cascade = cv2.CascadeClassifier()
        storage = None
        try:
            # OpenCV cannot open Unicode filenames reliably on Windows. Reading
            # the XML with Python also supports projects stored in Chinese paths.
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            cascade_xml = cascade_path.read_text(encoding="utf-8")
            storage = cv2.FileStorage(
                cascade_xml,
                cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY,
            )
            _cascade.read(storage.getFirstTopLevelNode())
        except (OSError, cv2.error):
            pass
        finally:
            if storage is not None:
                storage.release()
    return _cascade


def count_faces(image: np.ndarray) -> int:
    """Count faces in an image. Returns 0 if image is invalid."""
    if image is None or image.size == 0:
        return 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = _get_cascade()
    if cascade.empty():
        return 0
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces)


def has_too_many_faces(image: np.ndarray, max_faces: int = 3) -> bool:
    return count_faces(image) > max_faces
