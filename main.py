from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pathlib import Path
from PIL import Image
import time
import json
import shutil
import tempfile
from typing import List

from model import predict_batch
from utils import (
    save_upload_file,
    move_file_to_folder,
    unzip_to_temp,
    zip_folders,
    GOOD_FOLDER,
    BAD_FOLDER,
    UPLOAD_FOLDER
)

app = FastAPI(title="Eye Classification API (Images & ZIPs)")


# -----------------------------
# Cleanup Helper
# -----------------------------
def clean_uploads_except(file_to_keep: Path):

    for item in UPLOAD_FOLDER.iterdir():

        if item.resolve() == file_to_keep.resolve():
            continue

        try:
            if item.is_file():
                item.unlink()
            else:
                shutil.rmtree(item, ignore_errors=True)

        except Exception:
            pass


# -----------------------------
# Safe Image Loader
# -----------------------------
def load_image(path: Path):

    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


# -----------------------------
# API
# -----------------------------
@app.post("/upload_batch/")
async def upload_batch(files: List[UploadFile] = File(...)):

    start_total = time.time()

    summary = []

    temp_zip_folders = set()

    images = []
    image_paths = []

    temp_folders = []   # Track temp folders for cleanup


    # -----------------------------
    # Save & Read Files
    # -----------------------------
    for file in files:

        saved_path = save_upload_file(file, UPLOAD_FOLDER)


        # ZIP FILE
        if saved_path.suffix.lower() == ".zip":

            temp_folder = unzip_to_temp(saved_path)
            temp_folders.append(temp_folder)

            for img_file in temp_folder.rglob("*"):

                if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                    continue

                image = load_image(img_file)

                if image:
                    images.append(image)
                    image_paths.append(img_file)

            summary.append({
                "filename": file.filename,
                "type": "zip"
            })


        # NORMAL IMAGE
        else:

            image = load_image(saved_path)

            if image:
                images.append(image)
                image_paths.append(saved_path)

            summary.append({
                "filename": file.filename,
                "type": "image"
            })


    # -----------------------------
    # Batch Prediction
    # -----------------------------
    predictions = predict_batch(images)


    # -----------------------------
    # Move Images
    # -----------------------------
    for img_path, pred in zip(image_paths, predictions):

        target = GOOD_FOLDER if pred == 1 else BAD_FOLDER

        moved = move_file_to_folder(img_path, target)

        temp_zip_folders.add(target)

        summary.append({
            "image": img_path.name,
            "result": "Open Eyes 👁️" if pred == 1 else "Closed Eyes 😴",
            "saved_to": str(moved)
        })


    # -----------------------------
    # Save Summary
    # -----------------------------
    summary_folder = UPLOAD_FOLDER / f"summary_{int(time.time())}"
    summary_folder.mkdir(exist_ok=True)

    summary_path = summary_folder / "summary.json"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)


    temp_zip_folders.add(summary_folder)


    # -----------------------------
    # Create ZIP
    # -----------------------------
    combined_zip = UPLOAD_FOLDER / f"classified_batch_{int(time.time())}.zip"

    zip_folders(combined_zip, list(temp_zip_folders))


    # -----------------------------
    # Cleanup Temp Folders
    # -----------------------------
    for folder in temp_folders:
        shutil.rmtree(folder, ignore_errors=True)

    clean_uploads_except(combined_zip)


    end_total = time.time()

    print(
        f"Processed {len(images)} images in {end_total - start_total:.2f}s"
    )


    # -----------------------------
    # Return Result
    # -----------------------------
    return FileResponse(
        combined_zip,
        filename=combined_zip.name,
        media_type="application/zip"
    )
