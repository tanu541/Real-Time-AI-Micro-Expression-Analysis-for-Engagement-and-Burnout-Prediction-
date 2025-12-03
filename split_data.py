import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split

# Path to train folder
TRAIN_DIR = r"C:/Users/Asus/OneDrive/Desktop/CASME2/CASME2_Data/processed/Micro_Expressions/train"

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load entire training folder
full_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)

print("Total Images:", len(full_dataset))

# Generate indices
indices = list(range(len(full_dataset)))

# Split into 80% train, 20% validation
train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,
    shuffle=True,
    stratify=full_dataset.targets,
    random_state=42
)

# Create samplers
train_sampler = SubsetRandomSampler(train_idx)
val_sampler   = SubsetRandomSampler(val_idx)

# Create DataLoaders
train_loader = DataLoader(full_dataset, batch_size=32, sampler=train_sampler)
val_loader   = DataLoader(full_dataset, batch_size=32, sampler=val_sampler)

print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Data split successful!")
