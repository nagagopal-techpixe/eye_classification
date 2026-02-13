from pathlib import Path
import shutil
from fastapi import UploadFile
import zipfile
import tempfile
import uuid

BASE_DIR = Path.cwd()
GOOD_FOLDER = BASE_DIR / "good_images"
BAD_FOLDER = BASE_DIR / "bad_images"
UPLOAD_FOLDER = BASE_DIR / "uploads"

for folder in [GOOD_FOLDER, BAD_FOLDER, UPLOAD_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

def save_upload_file(upload_file: UploadFile, folder: Path) -> Path:
    ext = Path(upload_file.filename).suffix
    unique_name = f"{uuid.uuid4().hex}{ext}"
    path = folder / unique_name
    with open(path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer, length=1024 * 1024)
    return path

def move_file_to_folder(file_path: Path, target_folder: Path) -> Path:
    target_folder.mkdir(exist_ok=True)
    target_path = target_folder / file_path.name
    if target_path.exists():
        target_path = target_folder / f"{uuid.uuid4().hex}_{file_path.name}"
    shutil.move(str(file_path), str(target_path))
    return target_path

def unzip_to_temp(zip_path: Path) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="eye_"))
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)
    return temp_dir

def clean_uploads_folder():
    for item in UPLOAD_FOLDER.iterdir():
        try:
            if item.is_file(): item.unlink()
            else: shutil.rmtree(item)
        except: pass