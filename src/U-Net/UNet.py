import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3
)

