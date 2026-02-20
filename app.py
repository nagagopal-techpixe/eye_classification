import os
import time
import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, BackgroundTasks
import torch
from PIL import Image
from sklearn.cluster import DBSCAN
import open_clip
from io import BytesIO
import aiohttp
from typing import List, Optional
from pydantic import BaseModel

# ---------------- SETTINGS ---------------- #

EPS = 0.05
MIN_SAMPLES = 2
BATCH_SIZE = 16

# ---------------- APP ---------------- #

app = FastAPI(title="Wedding Photo Grouping & Quality Sorting")

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
    left = lms.landmark[33]
    right = lms.landmark[263]
    dx = right.x - left.x
    dy = right.y - left.y
    return abs(np.degrees(np.arctan2(dy, dx)))


def calculate_ear(lms, top, bottom, left, right):
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
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return []

    results = []
    for lms in result.multi_face_landmarks:
        l_ear = calculate_ear(lms, 159, 145, 33, 133)
        r_ear = calculate_ear(lms, 386, 374, 362, 263)
        ear = (l_ear + r_ear) / 2

        if ear < 0.18:
            eye_state = "closed"
            eye_score = 10
        elif ear < 0.23:
            eye_state = "half"
            eye_score = 50
        else:
            eye_state = "open"
            eye_score = 100

        tilt = get_face_tilt(lms)
        score = eye_score

        if tilt > 15:
            score *= 0.7
        if tilt > 25:
            score *= 0.4
        if tilt > 35:
            score *= 0.15

        results.append({
            "ear": round(ear, 3),
            "eye_state": eye_state,
            "tilt": round(tilt, 2),
            "score": round(score, 2)
        })

    return results


def is_blurry(image_pil):
    gray = np.array(image_pil.convert("L"))
    gray = cv2.resize(gray, (500, 500))
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    score = lap.var()
    return (score < 180), score


def get_image_quality_score(image_bytes):
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    is_blur, blur_score = is_blurry(pil_image)

    if is_blur:
        return "Bad", 0.0, 0, round(blur_score, 2)

    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return "Bad", 0.0, 0, round(blur_score, 2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(40, 40))

    all_scores = []
    all_states = []
    h, w, _ = img.shape

    if len(faces) > 0:
        for (x, y, bw, bh) in faces:
            pad = 0.15
            x1 = max(0, x - int(bw * pad))
            y1 = max(0, y - int(bh * pad))
            x2 = min(w, x + bw + int(bw * pad))
            y2 = min(h, y + bh + int(bh * pad))
            crop = img[y1:y2, x1:x2]
            for r in analyze_face(crop):
                all_scores.append(r["score"])
                all_states.append(r["eye_state"])

    if not all_scores:
        for r in analyze_face(img):
            all_scores.append(r["score"])
            all_states.append(r["eye_state"])

    people = len(all_scores)
    final_score = min(all_scores) if all_scores else 0

    if "closed" in all_states:
        final_score *= 0.3

    final_score = round(final_score, 2)
    status = "Good" if people >= 1 and final_score >= 50 else "Bad"

    return status, final_score, people, round(blur_score, 2)


# ---------------- CLUSTERING FUNCTIONS ---------------- #

def get_embeddings(images_bytes_list):
    tensors = [preprocess(Image.open(BytesIO(b)).convert("RGB")) for b in images_bytes_list]
    batch = torch.stack(tensors).to(device)

    with torch.no_grad():
        embeddings = model.encode_image(batch)

    embeddings = embeddings.cpu().numpy()
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


def cluster_images(filenames, embeddings):
    db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric="cosine")
    labels = db.fit_predict(embeddings)

    clusters = {}
    for file, label in zip(filenames, labels):
        label_key = "others" if label == -1 else int(label)
        clusters.setdefault(label_key, []).append(file)

    return clusters


# ---------------- MODELS ---------------- #

class ImageItem(BaseModel):
    imageId: str
    url: str


class ImageLinksRequest(BaseModel):
    images: List[ImageItem]
    callbackUrl: Optional[str] = None
    subEventId: Optional[str] = None
    store_clusters: Optional[bool] = False


# ---------------- API ENDPOINTS ---------------- #

@app.post("/process_image_links_stream")
async def process_image_links_stream(
    request: ImageLinksRequest,
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(process_images_and_callback, request)
    return {
        "status": "accepted",
        "message": "Processing started",
        "subEventId": request.subEventId
    }


async def process_images_and_callback(request: ImageLinksRequest):
    start_time = time.time()

    global_filenames = []
    global_embeddings = []
    global_image_quality = {}

    callback_url = request.callbackUrl
    sub_event_id = request.subEventId

    # 🔥 CHANGE: return FULL URL instead of filename
    async def fetch_image(session, url):
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    return url, data   # ← FULL URL stored here
        except Exception as e:
            print(f"Error fetching {url}: {e}")
        return None, None

    async with aiohttp.ClientSession() as session:
        for image in request.images:
            name, data = await fetch_image(session, image.url)

            if data is None:
                continue

            # Quality analysis
            status, score, people, blur_score = get_image_quality_score(data)

            global_image_quality[name] = {
                "status": status,
                "score": score,
                "people": people,
                "blur_score": blur_score
            }

            # Embedding generation
            embedding = get_embeddings([data])[0]
            global_embeddings.append(embedding)

            # Store FULL URL instead of filename
            global_filenames.append(name)

            del data

    # If no valid images processed
    if not global_embeddings:
        result_payload = {
            "subEventId": sub_event_id,
            "message": "No valid images processed",
            "total_images": 0,
            "groups": {},
            "blur": [],
            "processing_time_seconds": round(time.time() - start_time, 2)
        }
    else:
        all_embeddings = np.vstack(global_embeddings)
        clusters = cluster_images(global_filenames, all_embeddings)

        final_clusters = {}
        blur_list = []

        for label, files in clusters.items():
            sorted_files = sorted(
                files,
                key=lambda f: (
                    0 if global_image_quality[f]["status"] == "Good" else 1,
                    -global_image_quality[f]["score"]
                )
            )

            non_blurry_files = []
            for f in sorted_files:
                if global_image_quality[f]["blur_score"] < 180:
                    blur_list.append({
                        "filename": f,  # ← now FULL URL
                        "blur_score": global_image_quality[f]["blur_score"],
                        "status": "Blurry"
                    })
                else:
                    non_blurry_files.append(f)

            if non_blurry_files:
                final_clusters[label] = non_blurry_files

        result_payload = {
            "subEventId": sub_event_id,
            "message": "Processing completed",
            "total_images": len(global_filenames),
            "groups": final_clusters,   # ← FULL URLs here
            "blur": blur_list,          # ← FULL URLs here
            "processing_time_seconds": round(time.time() - start_time, 2)
        }

    # Send callback
    if callback_url:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(callback_url, json=result_payload) as resp:
                    print(f"Callback status: {resp.status}")
        except Exception as e:
            print(f"Callback failed: {e}")

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