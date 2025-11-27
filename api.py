# api.py – FINAL FastAPI Backend for Hand Gesture Classification

import io
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

CHECKPOINT_PATH = "best_efficientnet_gesture.pth"   # final trained model file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# FASTAPI Setup
# ------------------------------------------------------------

app = FastAPI()

# Allow all frontend domains (Lovable / HTML / React etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# Image Transform (same as validation transform)
# ------------------------------------------------------------

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# ------------------------------------------------------------
# Load Model + Class Names
# ------------------------------------------------------------

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
class_names = checkpoint["class_names"]

weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights)

# Replace classifier to match number of classes
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, len(class_names))

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print("🔥 Model Loaded & Ready!")

# ------------------------------------------------------------
# Image Preprocessing Function
# ------------------------------------------------------------

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(img).unsqueeze(0)

# ------------------------------------------------------------
# API ROUTES
# ------------------------------------------------------------

@app.get("/")
async def home():
    return {"status": "Hand Gesture API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        input_tensor = preprocess_image(image_bytes).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

        predicted_class = class_names[pred_idx.item()]
        confidence = float(conf.item())

        return {
            "predicted_class": predicted_class,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}
