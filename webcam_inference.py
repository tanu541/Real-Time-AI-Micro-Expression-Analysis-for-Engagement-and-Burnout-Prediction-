import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- Classes ----------------
VALID_CLASSES = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

# ---------------- Load Model ----------------
num_classes = len(VALID_CLASSES)
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)

MODEL_PATH = "best_model.pth"
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ---------------- Transform ----------------
transform = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---------------- Webcam ----------------
cap = cv2.VideoCapture(0)  # default webcam

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR (OpenCV) to RGB (PIL)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Model prediction
    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)
        label = VALID_CLASSES[pred.item()]

    # Display label
    cv2.putText(frame, f"Expression: {label}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

    cv2.imshow("Micro-Expression Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
