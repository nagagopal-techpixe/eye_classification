import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification


# -----------------------------
# Device (CPU / GPU)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)


# -----------------------------
# Load Model (Once Only)
# -----------------------------
MODEL_FOLDER = "eye_model"

model = AutoModelForImageClassification.from_pretrained(MODEL_FOLDER)
model.to(device)
model.eval()

print("Model loaded ✅")


# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# Single Image Prediction
# (Keep for compatibility)
# -----------------------------
def predict_image(image: Image.Image) -> int:
    """
    Return:
    0 = Closed Eyes
    1 = Open Eyes
    """

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output.logits, dim=1).item()

    return pred


# -----------------------------
# Batch Prediction (FAST 🚀)
# -----------------------------
def predict_batch(images: list[Image.Image], batch_size: int = 32):
    """
    Predict multiple images together (Much Faster)

    Returns:
    List of predictions (0 or 1)
    """

    results = []

    for i in range(0, len(images), batch_size):

        batch = images[i:i + batch_size]

        # Convert all images to tensors
        tensors = torch.stack([
            transform(img) for img in batch
        ]).to(device)

        with torch.no_grad():
            outputs = model(tensors)
            preds = torch.argmax(outputs.logits, dim=1)

        results.extend(preds.cpu().tolist())

    return results
