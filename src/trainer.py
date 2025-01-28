import math
import os

import torch.nn as nn
import AwakeNet
import torch
import net_params
device = net_params.device

criterion = nn.BCELoss()

def train_one_batch(net : AwakeNet, batched_real_img : torch.Tensor, batched_dreamt_img : torch.Tensor):
    real_scores, fake_scores_d, fake_scores_g = net(batched_real_img, batched_dreamt_img)

    net.generator.optim.zero_grad()

    g_loss = criterion(fake_scores_g, torch.ones_like(fake_scores_g))
    print(f"generator loss : {g_loss}")
    g_loss.backward()
    net.generator.optim.step()

    for dis in net.discriminators:
        dis.optim.zero_grad()

    d_real_loss = criterion(real_scores, torch.ones_like(real_scores))
    d_fake_loss = criterion(fake_scores_d, torch.zeros_like(fake_scores_d))

    d_loss = torch.add(d_real_loss, d_fake_loss)
    print(f"discriminator loss : {d_loss}")
    d_loss.backward()
    for dis in net.discriminators:
        dis.optim.step()


def train(net : AwakeNet, real_imgs : torch.Tensor, fake_dreamt_img : torch.Tensor):
    for i in range(net_params.batch_size):
        print(f"batch {i + 1} start training")
        whole_size = real_imgs.shape[0]
        if whole_size <= net_params.batch_size:
            train_one_batch(net, real_imgs, fake_dreamt_img)
        else:
            for i in range(math.ceil(whole_size / net_params.batch_size)):
                train_one_batch(
                    net,
                    real_imgs[i * net_params.batch_size: min((i + 1) * net_params.batch_size - 1, whole_size)],
                    fake_dreamt_img[i * net_params.batch_size: min((i + 1) * net_params.batch_size - 1, whole_size)])

## test
print(device)
train_net = AwakeNet.AwakeNet(n_dis =1)
train_net.initial()
train_net.to(device)
train_net.train()

''' LOAD DATA  '''

real_img = []
dream_img = []


import cv2 as cv
print("start loading data")
r_fold = os.listdir(r"./raw_data")
for r in r_fold:
    di = cv.resize(cv.imread(r"./dream_data/" + r), [net_params.input_image_size_x, net_params.input_image_size_y])
    ri = cv.resize(cv.imread(r"./raw_data/" + r), [net_params.input_image_size_x, net_params.input_image_size_y])
    dream_img.append(torch.tensor(di \
                                  .reshape((net_params.input_image_depth,
                                           net_params.input_image_size_x,
                                           net_params.input_image_size_y)),dtype=torch.float32))
    real_img.append(torch.tensor(ri\
                                 .reshape((net_params.input_image_depth,
                                           net_params.input_image_size_x,
                                           net_params.input_image_size_y)),dtype=torch.float32))

r_datas = torch.stack(real_img,dim=0)
f_datas = torch.stack(dream_img, dim=0)
print("end loading data")
print("start training")
train(train_net, r_datas.to(device), f_datas.to(device))

