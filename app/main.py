from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import logging
import os

logging.basicConfig(
    filename="api_requests.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI(
    title="Cats vs Dogs Prediction API",
    description="Predicts whether an uploaded image is a cat or a dog",
    version="1.0.0"
)

CLASS_NAMES = ["cat", "dog"]
MODEL_PATH = os.environ.get("MODEL_PATH", "catsdogs_model.pt")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = SimpleCNN(num_classes=2)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

METRICS = {"request_count": 0, "total_latency_ms": 0.0}


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.get("/", response_model=HealthResponse)
def home():
    return HealthResponse(status="running", model_loaded=os.path.exists(MODEL_PATH))


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=os.path.exists(MODEL_PATH))


@app.get("/metrics")
def metrics():
    count = METRICS["request_count"]
    avg_latency = (METRICS["total_latency_ms"] / count) if count else 0.0
    return {"request_count": count, "avg_latency_ms": round(avg_latency, 2)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    import time
    start = time.time()

    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Upload a JPEG or PNG image.")

    try:
        image_bytes = await file.read()
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        tensor = TRANSFORM(img).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).tolist()

        label = CLASS_NAMES[int(torch.tensor(probs).argmax())]
        result = {
            "label": label,
            "probabilities": {cls: round(p, 4) for cls, p in zip(CLASS_NAMES, probs)},
        }

        latency_ms = (time.time() - start) * 1000
        METRICS["request_count"] += 1
        METRICS["total_latency_ms"] += latency_ms

        logging.info(f"filename={file.filename} label={label} latency_ms={latency_ms:.1f}")
        return result

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    # demo run for screen recording