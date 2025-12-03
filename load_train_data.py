import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

TRAIN_DIR = r"C:/Users/Asus/OneDrive/Desktop/CASME2/CASME2_Data/processed/Micro_Expressions/train"


# Transformations for preprocessing images
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# Load ONLY the train folder
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)

# Check classes found automatically from subfolders
print("Classes:", train_dataset.classes)
print("Number of training images:", len(train_dataset))

# Create DataLoader (for training)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print("Training folder loaded successfully!")
