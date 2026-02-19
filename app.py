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
from pydantic import BaseModel
import aiohttp
import asyncio

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

class ImageLinksRequest(BaseModel):
    urls: List[str]  # List of image URLs
    store_clusters: bool = False
@app.post("/process_image_links_stream")
async def process_image_links_stream(request: ImageLinksRequest):
    import aiohttp
    start_time = time.time()

    global_filenames = []
    global_embeddings = []
    global_image_quality = {}

    async def fetch_image(session, url):
        async with session.get(url) as resp:
            if resp.status == 200:
                data = await resp.read()
                name = url.split("/")[-1]
                return name, data
            return None, None

    async with aiohttp.ClientSession() as session:
        for idx, url in enumerate(request.urls, start=1):
            name, data = await fetch_image(session, url)
            if data is None:
                print(f"Skipping {url}, failed to download")
                continue

            # --- Analyze quality ---
            status, score, people, blur_score = get_image_quality_score(data)
            global_image_quality[name] = {
                "status": status,
                "score": score,
                "people": people,
                "blur_score": blur_score
            }

            # --- Compute embedding ---
            embedding = get_embeddings([data])[0]  # single image
            global_embeddings.append(embedding)
            global_filenames.append(name)

            # --- Clear image bytes from memory ---
            del data

            print(f"Processed {idx}/{len(request.urls)}: {name} -> {status}, score={score}")

    # --- Global clustering ---
    all_embeddings = np.vstack(global_embeddings)
    clusters = cluster_images(global_filenames, all_embeddings)

    # --- Sort within clusters by quality ---
    final_clusters = {}
    blur_list = []
    for label, files in clusters.items():
        sorted_files = sorted(
            files,
            key=lambda f: (0 if global_image_quality[f]["status"] == "Good" else 1,
                           -global_image_quality[f]["score"])
        )
        non_blurry_files = []
        for f in sorted_files:
            if global_image_quality[f]["blur_score"] < 180:
                blur_list.append({
                    "filename": f,
                    "blur_score": global_image_quality[f]["blur_score"],
                    "status": "Blurry"
                })
            else:
                non_blurry_files.append(f)
        if non_blurry_files:
            final_clusters[label] = non_blurry_files

    processing_time = round(time.time() - start_time, 2)
    return {
        "message": "Global clustering done successfully",
        "total_images": len(global_filenames),
        "groups": final_clusters,
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


    