import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")
        self.image_filenames = sorted([
        f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))
    ])
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", ".png"))

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")  
       
        mask = Image.open(mask_path)
        mask = transforms.Resize((128, 128), interpolation=Image.NEAREST)(mask)
        mask = np.array(mask, dtype=np.uint8) // 255
        mask = torch.tensor(mask, dtype=torch.long) 

        if self.transform:
            image = self.transform(image)

        return image, mask
