import io
import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware


ONNX_PATH = "models\stage_classifier.onnx" 
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


session = ort.InferenceSession(ONNX_PATH)
input_name = session.get_inputs()[0].name

def preprocess_image(image: Image.Image) -> np.ndarray:
    
    image = image.resize((224, 224))
    
    img_data = np.array(image).astype(np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std
    
    img_data = np.transpose(img_data, (2, 0, 1))
    
    img_data = np.expand_dims(img_data, axis=0)
    
    return img_data.astype(np.float32)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1)

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    
    input_tensor = preprocess_image(image)

    
    outputs = session.run(None, {input_name: input_tensor})
    logits = outputs[0]
    
    
    probabilities = softmax(logits)
    predicted_class = np.argmax(probabilities, axis=1)[0]
    confidence = probabilities[0][predicted_class] * 100

    return {
        "predicted_stage": STAGE_LABELS[predicted_class],
        "confidence": f"{confidence:.1f}%"
    }