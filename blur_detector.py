import cv2
import numpy as np
from PIL import Image


def is_blurry(image: Image.Image, threshold: float = 100.0):
    """
    Returns:
    True  = Blurry
    False = Clear
    """

    # Convert PIL → OpenCV
    img = np.array(image.convert("L"))

    # Laplacian variance
    fm = cv2.Laplacian(img, cv2.CV_64F).var()

    return fm < threshold, fm
