import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split

# ---------------- Paths ----------------
TRAIN_DIR = r"C:\Users\Asus\OneDrive\Desktop\CASME2\CASME2_Data\processed\Micro_Expressions\train"

# ---------------- Transform ----------------
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ---------------- Custom Dataset ----------------
class ImageFolderWithFilter(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        self.classes = []

        # get all class folders
        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                self.classes.append(class_name)
                for file_name in os.listdir(class_path):
                    file_path = os.path.join(class_path, file_name)
                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                        self.samples.append((file_path, len(self.classes)-1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# ---------------- Load Dataset ----------------
full_dataset = ImageFolderWithFilter(TRAIN_DIR, transform=transform)
print("Total images:", len(full_dataset))
print("Classes:", full_dataset.classes)

# ---------------- Split Train/Validation ----------------
indices = list(range(len(full_dataset)))
train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,
    shuffle=True,
    stratify=[label for _, label in full_dataset.samples],
    random_state=42
)

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

# ---------------- DataLoaders ----------------
batch_size = 32
train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0, pin_memory=True)
val_loader   = DataLoader(full_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=0, pin_memory=True)

print("Training DataLoader and Validation DataLoader created!")
print("Number of training batches:", len(train_loader))
print("Number of validation batches:", len(val_loader))

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- Model ----------------
num_classes = len(full_dataset.classes)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Windows-safe
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# ---------------- Loss & Optimizer ----------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ---------------- Training Loop ----------------
num_epochs = 10
best_val_acc = 0.0

for epoch in range(num_epochs):
    # ---- TRAIN ----
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_acc = 100. * correct / total
    train_loss = train_loss / total

    # ---- VALIDATION ----
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_acc = 100. * correct / total
    val_loss = val_loss / total

    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved Best Model!")
