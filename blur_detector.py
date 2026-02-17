from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np
import io

import cv2   # make sure installed

<<<<<<< HEAD
def is_blurry(image: Image.Image):
    """Calculates Laplacian variance to detect edge sharpness."""
    # Convert PIL to grayscale numpy
    gray = np.array(image.convert("L"))
=======

app = FastAPI(title="Blur Detection API")


# --------------------------------
# Blur Detection
# --------------------------------
def is_blurry(image: Image.Image):

    # Convert PIL to grayscale numpy
    gray = np.array(image.convert("L"))

    # Resize for stability
>>>>>>> 509a6187c77edef232aa143f3deaea6a7ca9b131
    gray = cv2.resize(gray, (500, 500))

    # Laplacian variance
    lap = cv2.Laplacian(gray, cv2.CV_64F)
<<<<<<< HEAD
    score = lap.var()

    threshold = 180
    return (score < threshold), score
=======

    score = lap.var()

    threshold = 180

    blurry = score < threshold

    return blurry, score


# --------------------------------
# API
# --------------------------------
@app.post("/check_blur/")
async def check_blur(file: UploadFile = File(...)):

    try:

        contents = await file.read()

        if not contents:
            return {"error": "Empty file"}

        image = Image.open(io.BytesIO(contents)).convert("RGB")

        blurry, score = is_blurry(image)

        return {
            "filename": file.filename,
            "blurry": bool(blurry),
            "blur_score": round(float(score), 2),
            "status": "Blurry ❌" if blurry else "Clear ✅"
        }

    except Exception as e:

        # Print error in terminal
        print("ERROR:", e)

        return {
            "error": str(e)
        }
>>>>>>> 509a6187c77edef232aa143f3deaea6a7ca9b131
