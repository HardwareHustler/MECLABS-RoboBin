import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys

# ---- CONFIG ----
MODEL_PATH = "waste_classifier.pth"
IMAGE_SIZE = 224
CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# ---- DEVICE ----
device = torch.device("cpu")

# ---- LOAD MODEL ----
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ---- IMAGE TRANSFORM ----
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return CLASS_NAMES[predicted.item()], confidence.item()

# ---- MAIN ----
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python infer.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    label, conf = predict(image_path)

    print(f"🗑️ Prediction: {label}")
    print(f"📊 Confidence: {conf:.2f}")
