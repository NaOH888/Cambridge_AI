from torch import nn
import torch.optim as optim
import torch
from dataload import *
import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet.model.to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 500

last_loss = 1000

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as t:

        for dream_imgs, real_imgs in t:
            dream_imgs, real_imgs = dream_imgs.to(device), real_imgs.to(device)

            # forward
            outputs = model(dream_imgs)
            loss = criterion(outputs, real_imgs)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
    if last_loss > epoch_loss:
        last_loss = epoch_loss
        print(f'Epoch {epoch+1}/{num_epochs}, AverageLoss: {epoch_loss/len(train_loader):.4f}')
        torch.save(model.state_dict(), 'dream2real.pth')
    else:
        print(f'Epoch {epoch+1}/{num_epochs}, AverageLoss: {epoch_loss/len(train_loader):.4f}')
