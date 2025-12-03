import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
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
