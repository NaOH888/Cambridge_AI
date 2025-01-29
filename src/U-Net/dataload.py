import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


class DreamToRealDataset(Dataset):
    def __init__(self, dream_dir, real_dir, transform=None):
        self.dream_images = sorted([os.path.join(dream_dir, f) for f in os.listdir(dream_dir) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')])
        self.real_images = sorted([os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')])
        self.transform = transform

    def __len__(self):
        return len(self.dream_images)
    
    def __getitem__(self, idx):
        dream_img = Image.open(self.dream_images[idx]).convert("RGB")
        real_img = Image.open(self.real_images[idx]).convert("RGB")

        if self.transform:
            dream_img = self.transform(dream_img)
            real_img = self.transform(real_img)
        
        return dream_img, real_img
    
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print('Loading data...')
train_dataset = DreamToRealDataset('src/dream_data', 'src/raw_data', transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
print('start training')