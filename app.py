import os
import time
import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, UploadFile, File
from typing import List
import torch
from PIL import Image
from sklearn.cluster import DBSCAN
import open_clip
from io import BytesIO
from tqdm import tqdm
import zipfile

# ---------------- SETTINGS ---------------- #

OUTPUT_FOLDER = "grouped"
BLUR_FOLDER = "blurry_rejected"
EPS = 0.05       # similarity threshold
MIN_SAMPLES = 2   # min images per group
BATCH_SIZE = 16   # number of images per batch for GPU


# ---------------- APP ---------------- #

app = FastAPI(title="Wedding Photo Grouping & Quality Sorting")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(BLUR_FOLDER, exist_ok=True)


# ---------------- OPENCLIP MODEL ---------------- #

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', 
    pretrained='laion2b_s34b_b79k'
)
model.to(device)
model.eval()


# ---------------- FACE DETECTION MODELS ---------------- #

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=30,
    refine_landmarks=True,
    min_detection_confidence=0.5
)


# ---------------- FACE QUALITY UTILS ---------------- #

def get_face_tilt(lms):
    """Calculate face tilt angle."""
    left = lms.landmark[33]
    right = lms.landmark[263]
    
    dx = right.x - left.x
    dy = right.y - left.y
    
    angle = np.degrees(np.arctan2(dy, dx))
    
    return abs(angle)


def calculate_ear(lms, top, bottom, left, right):
    """Calculate Eye Aspect Ratio."""
    v = np.linalg.norm([
        lms.landmark[top].x - lms.landmark[bottom].x,
        lms.landmark[top].y - lms.landmark[bottom].y
    ])
    
    h = np.linalg.norm([
        lms.landmark[left].x - lms.landmark[right].x,
        lms.landmark[left].y - lms.landmark[right].y
    ])
    
    return v / (h + 1e-6)


def analyze_face(img):
    """Analyze face quality metrics."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    
    if not result.multi_face_landmarks:
        return []
    
    results = []
    
    for lms in result.multi_face_landmarks:
        # Calculate EAR for both eyes
        l_ear = calculate_ear(lms, 159, 145, 33, 133)
        r_ear = calculate_ear(lms, 386, 374, 362, 263)
        ear = (l_ear + r_ear) / 2
        
        # Determine eye state
        if ear < 0.18:
            eye_state = "closed"
            eye_score = 10
        elif ear < 0.23:
            eye_state = "half"
            eye_score = 50
        else:
            eye_state = "open"
            eye_score = 100
        
        # Calculate face tilt
        tilt = get_face_tilt(lms)
        
        # Calculate score
        score = eye_score
        
        # Penalize tilt
        if tilt > 15:
            score *= 0.7
        if tilt > 25:
            score *= 0.4
        if tilt > 35:
            score *= 0.15
        
        score = round(score, 2)
        
        results.append({
            "ear": round(ear, 3),
            "eye_state": eye_state,
            "tilt": round(tilt, 2),
            "score": score
        })
    
    return results


def is_blurry(image_pil):
    """Calculates Laplacian variance to detect edge sharpness."""
    # Convert PIL to grayscale numpy
    gray = np.array(image_pil.convert("L"))
    gray = cv2.resize(gray, (500, 500))

    # Laplacian variance
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    score = lap.var()

    threshold = 180
    return (score < threshold), score


def get_image_quality_score(image_bytes):
    """
    Analyze image quality and return status and score.
    Returns: (status: str, final_score: float, people: int, blur_score: float)
    """
    # Convert bytes to PIL image for blur check
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    
    # Check blur first - reject immediately if blurry
    is_blur, blur_score = is_blurry(pil_image)
    
    if is_blur:
        return "Bad", 0.0, 0, round(blur_score, 2)
    
    # Convert bytes to cv2 image for face analysis
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img is None:
        return "Bad", 0.0, 0, round(blur_score, 2)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        1.1,
        3,
        minSize=(40, 40)
    )
    
    all_scores = []
    all_states = []
    
    h, w, _ = img.shape
    
    # Analyze face crops
    if len(faces) > 0:
        for (x, y, bw, bh) in faces:
            pad = 0.15
            
            x1 = max(0, x - int(bw * pad))
            y1 = max(0, y - int(bh * pad))
            x2 = min(w, x + bw + int(bw * pad))
            y2 = min(h, y + bh + int(bh * pad))
            
            crop = img[y1:y2, x1:x2]
            results = analyze_face(crop)
            
            for r in results:
                all_scores.append(r["score"])
                all_states.append(r["eye_state"])
    
    # Fallback to full image if no faces detected
    if not all_scores:
        results = analyze_face(img)
        for r in results:
            all_scores.append(r["score"])
            all_states.append(r["eye_state"])
    
    # Calculate final decision
    people = len(all_scores)
    final_score = min(all_scores) if all_scores else 0
    
    # Reject if ANY eyes closed
    if "closed" in all_states:
        final_score *= 0.3
    
    final_score = round(final_score, 2)
    
    # Determine status
    if people >= 1 and final_score >= 50:
        status = "Good"
    else:
        status = "Bad"
    
    return status, final_score, people, round(blur_score, 2)


# ---------------- CLUSTERING FUNCTIONS ---------------- #

def get_embeddings(images_bytes_list):
    """Compute embeddings for a list of image bytes (batch)."""
    tensors = [preprocess(Image.open(BytesIO(b)).convert("RGB")) for b in images_bytes_list]
    batch = torch.stack(tensors).to(device)
    
    with torch.no_grad():
        embeddings = model.encode_image(batch)
    
    embeddings = embeddings.cpu().numpy()
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    return embeddings


def cluster_images(filenames, embeddings):
    """Cluster embeddings using DBSCAN."""
    db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric="cosine")
    labels = db.fit_predict(embeddings)
    
    clusters = {}
    for file, label in zip(filenames, labels):
        if label == -1:
            label_key = "others"
        else:
            label_key = int(label)
        clusters.setdefault(label_key, []).append(file)
    
    return clusters


# ---------------- API ENDPOINT ---------------- #



@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...), store_clusters: bool = False):
    """
    Upload images or zip files, group them, analyze quality, and sort within groups.
    Good images appear first in each group.
    """
    start_time = time.time()

    # List to store all images
    image_bytes_list = []
    filenames = []

    # --- 1. Process each uploaded file ---
    for file in files:
        content = await file.read()
        name = file.filename.lower()

        # If it's a zip file, extract images
        if name.endswith(".zip"):
            with zipfile.ZipFile(BytesIO(content)) as z:
                for info in z.infolist():
                    if info.filename.lower().endswith((".jpg", ".jpeg", ".png")):
                        with z.open(info) as img_file:
                            image_bytes = img_file.read()
                            image_bytes_list.append(image_bytes)
                            filenames.append(info.filename)
        else:
            # Regular image file
            image_bytes_list.append(content)
            filenames.append(file.filename)

    print(f"Loaded {len(filenames)} images (from individual files + zips)")

    # --- 2. Compute embeddings in batches ---
    embeddings_list = []
    for i in range(0, len(image_bytes_list), BATCH_SIZE):
        batch_bytes = image_bytes_list[i:i + BATCH_SIZE]
        batch_embeddings = get_embeddings(batch_bytes)
        embeddings_list.append(batch_embeddings)

    embeddings = np.vstack(embeddings_list)

    # --- 3. Cluster images ---
    clusters = cluster_images(filenames, embeddings)

    # --- 4. Analyze image quality ---
    print("Analyzing image quality...")
    image_quality = {}

    for filename, img_bytes in zip(filenames, image_bytes_list):
        status, score, people, blur_score = get_image_quality_score(img_bytes)
        image_quality[filename] = {
            "status": status,
            "score": score,
            "people": people,
            "blur_score": blur_score
        }

    # --- 5. Sort clusters ---
    sorted_clusters = {}
    for label, files_in_group in clusters.items():
        sorted_files = sorted(
            files_in_group,
            key=lambda f: (
                0 if image_quality[f]["status"] == "Good" else 1,
                -image_quality[f]["score"]
            )
        )
        sorted_clusters[label] = sorted_files

    # --- 6. Save clustered images ---
    if store_clusters:
        for label, sorted_files in sorted_clusters.items():
            folder_name = f"group_{label}" if label != "others" else "others"
            folder_path = os.path.join(OUTPUT_FOLDER, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            for idx, filename in enumerate(sorted_files):
                file_idx = filenames.index(filename)
                img = Image.open(BytesIO(image_bytes_list[file_idx])).convert("RGB")
                quality_info = image_quality[filename]

                if quality_info["blur_score"] < 180:
                    # Use only the basename so we don't recreate nested folders
                    base_name = os.path.basename(filename)
                    blur_filename = f"blur_{quality_info['blur_score']:.1f}_{base_name}"
                    blur_path = os.path.join(BLUR_FOLDER, blur_filename)
                    os.makedirs(BLUR_FOLDER, exist_ok=True)
                    # Avoid overwriting existing files by appending a counter
                    bname, bext = os.path.splitext(blur_filename)
                    counter = 1
                    while os.path.exists(blur_path):
                        blur_filename = f"{bname}_{counter}{bext}"
                        blur_path = os.path.join(BLUR_FOLDER, blur_filename)
                        counter += 1
                    img.save(blur_path)
                else:
                    prefix = f"{idx:03d}_{quality_info['status']}_"
                    # Flatten any nested paths - keep only filename
                    safe_name = os.path.basename(filename)
                    new_filename = prefix + safe_name
                    # Ensure group folder exists
                    os.makedirs(folder_path, exist_ok=True)
                    # Avoid overwriting by adding a numeric suffix when needed
                    base_name, ext = os.path.splitext(new_filename)
                    candidate = new_filename
                    counter = 1
                    while os.path.exists(os.path.join(folder_path, candidate)):
                        candidate = f"{base_name}_{counter}{ext}"
                        counter += 1
                    target_path = os.path.join(folder_path, candidate)
                    img.save(target_path)

    # --- 7. Prepare response ---
    groups_with_quality = {}
    blur_list = []

    for label, sorted_files in sorted_clusters.items():
        non_blurry_files = []

        for f in sorted_files:
            if image_quality[f]["blur_score"] < 180:
                blur_list.append({
                    "filename": f,
                    "blur_score": image_quality[f]["blur_score"],
                    "status": "Blurry"
                })
            else:
                non_blurry_files.append(f)

        if non_blurry_files:
            groups_with_quality[label] = non_blurry_files

    blurry_count = len(blur_list)
    good_count = sum(1 for q in image_quality.values() if q["status"] == "Good" and q["blur_score"] >= 180)
    bad_count = sum(1 for q in image_quality.values() if q["status"] == "Bad" and q["blur_score"] >= 180)

    end_time = time.time()
    processing_time = round(end_time - start_time, 2)

    return {
        "message": "Images grouped, analyzed, and sorted successfully",
        "total_images": len(filenames),
        # "total_groups": len(groups_with_quality),
        "groups": groups_with_quality,
        "blur": blur_list,
        "processing_time_seconds": processing_time
    }


@app.get("/")
def root():
    return {
        "message": "Wedding Photo Grouping & Quality Sorting API is running",
        "features": [
            "Automatic image grouping by similarity",
            "Blur detection (blurry images separated)",
            "Quality analysis (eye state, face tilt)",
            "Sorted output (Good images first)"
        ]
    }


        