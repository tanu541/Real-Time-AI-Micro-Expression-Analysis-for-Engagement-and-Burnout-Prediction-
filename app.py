# file: app.py
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import torch
from torchvision import transforms, models
from torchvision.models import mobilenet_v2


app = FastAPI()

# Mount static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---------------- Classes ----------------
VALID_CLASSES = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Load Model ----------------
num_classes = len(VALID_CLASSES)
model = models.resnet18(weights=None)  # or MobileNetV2 for faster inference
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("best_model_updated.pth", map_location=device))
model = model.to(device)
model.eval()

# ---------------- Transform ----------------
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# ---------------- Routes ----------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Open image
    img = Image.open(file.file).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Inference with no_grad
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probabilities, 1)
        label = VALID_CLASSES[pred_idx.item()]

    return {
        "prediction": label,
        "confidence": round(confidence.item() * 100, 2)  # return as percentage
    }
