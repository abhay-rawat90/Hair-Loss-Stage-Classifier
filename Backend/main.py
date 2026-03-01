import io
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware


MODEL_PATH = "models/stage_classifier.pth"
NUM_CLASSES = 7

STAGE_LABELS = {
    0: "Stage 1 - No Hair Loss",
    1: "Stage 2 - Slight Recession",
    2: "Stage 3 - Visible Thinning",
    3: "Stage 4 - Significant Loss",
    4: "Stage 5 - Severe Loss",
    5: "Stage 6 - Very Severe",
    6: "Stage 7 - Extreme Baldness"
}


app = FastAPI(title="Hair Loss Stage Classifier API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = models.resnet50(pretrained=False)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, NUM_CLASSES)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    stage = STAGE_LABELS[predicted_class.item()]
    conf = confidence.item() * 100

    return {
        "predicted_stage": stage,
        "confidence": f"{conf:.1f}%"
    }