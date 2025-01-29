import torch
from torch import nn
from tqdm import tqdm
import os
import UNet
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), 
])

model = UNet.model

model = model.to(device)

model.load_state_dict(torch.load("src\\U-Net\\dream2real.pth", map_location=device))  
model.eval()  

def process_images_in_folder(input_folder, output_folder):

    img_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    os.makedirs(output_folder, exist_ok=True)
    
    for img_file in img_files:

        img_path = os.path.join(input_folder, img_file)
        
        input_image = Image.open(img_path).convert("RGB")  
        
        input_tensor = transform(input_image).unsqueeze(0).to(device)
        
        with torch.no_grad():  
            output = model(input_tensor) 
            output = output.squeeze(0)   
            output = output.cpu().numpy()  

        output_image = output.transpose(1, 2, 0)  
        output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())  # 归一化到 [0, 1]
        
        output_img_path = os.path.join(output_folder, img_file)
        plt.imsave(output_img_path, output_image)  

        print(f"Processed: {img_file}")


input_folder = "src\\U-Net\\input"  # 输入文件夹路径
output_folder = "src\\U-Net"  # 输出文件夹路径
process_images_in_folder(input_folder, output_folder)
