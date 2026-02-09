from pathlib import Path
import shutil
from fastapi import UploadFile
import zipfile
import tempfile
import uuid


# -----------------------------
# Folder paths
# -----------------------------
BASE_DIR = Path.cwd()

GOOD_FOLDER = BASE_DIR / "good_images"
BAD_FOLDER = BASE_DIR / "bad_images"
UPLOAD_FOLDER = BASE_DIR / "uploads"


# -----------------------------
# Create folders
# -----------------------------
for folder in [GOOD_FOLDER, BAD_FOLDER, UPLOAD_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Save Uploaded File (Fast + Safe)
# -----------------------------
def save_upload_file(upload_file: UploadFile, folder: Path) -> Path:
    """
    Save uploaded file with unique name
    (Avoid overwrite in production)
    """

    ext = Path(upload_file.filename).suffix
    unique_name = f"{uuid.uuid4().hex}{ext}"

    path = folder / unique_name

    with open(path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer, length=1024 * 1024)

    return path


# -----------------------------
# Move File
# -----------------------------
def move_file_to_folder(file_path: Path, target_folder: Path) -> Path:

    target_folder.mkdir(exist_ok=True)

    target_path = target_folder / file_path.name

    if target_path.exists():
        target_path = target_folder / f"{uuid.uuid4().hex}_{file_path.name}"

    shutil.move(str(file_path), str(target_path))

    return target_path


# -----------------------------
# Unzip Faster
# -----------------------------
def unzip_to_temp(zip_path: Path) -> Path:

    temp_dir = Path(tempfile.mkdtemp(prefix="eye_"))

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    return temp_dir


# -----------------------------
# Zip Multiple Folders (Fast)
# -----------------------------
def zip_folders(output_zip: Path, folders: list[Path]):

    with zipfile.ZipFile(
        output_zip,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6
    ) as zipf:

        for folder in folders:

            if not folder.exists():
                continue

            for file in folder.rglob("*"):

                if file.is_file():

                    arcname = file.relative_to(folder.parent)

                    zipf.write(file, arcname=arcname)

    return output_zip


# -----------------------------
# Cleanup
# -----------------------------
def clean_temp_folder(folder: Path):

    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)


def clean_uploads_folder():

    if not UPLOAD_FOLDER.exists():
        return

    for item in UPLOAD_FOLDER.iterdir():

        try:
            if item.is_file():
                item.unlink()
            else:
                shutil.rmtree(item)

        except Exception as e:
            print("Cleanup error:", e)
