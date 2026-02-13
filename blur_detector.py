import cv2
import numpy as np
from PIL import Image

def is_blurry(image: Image.Image):
    """Calculates Laplacian variance to detect edge sharpness."""
    # Convert PIL to grayscale numpy
    gray = np.array(image.convert("L"))
    gray = cv2.resize(gray, (500, 500))

    # Laplacian variance
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    score = lap.var()

    threshold = 180
    return (score < threshold), score