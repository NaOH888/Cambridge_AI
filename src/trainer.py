import math
import os

import torch.nn as nn
from torch import dtype

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
            for j in range(math.ceil(whole_size / net_params.batch_size)):
                train_one_batch(
                    net,
                    real_imgs[j * net_params.batch_size: min((j + 1) * net_params.batch_size - 1, whole_size)],
                    fake_dreamt_img[j * net_params.batch_size: min((j + 1) * net_params.batch_size - 1, whole_size)])

def pretrain_for_generator(net : AwakeNet, real_imgs, dream_imgs, epoches):
    gen = net.generator
    whole_size = real_imgs.shape[0]
    t_criterion = nn.MSELoss()
    for epoch in range(epoches):
        net.saveParam()
        print("model saved")
        print(f"epoch {epoch + 1} start training")
        total_loss = 0
        blocks = math.ceil(whole_size / net_params.batch_size)
        for i in range(blocks):

            t_r_i = real_imgs[i * net_params.batch_size: min((i + 1) * net_params.batch_size - 1, whole_size)]
            t_d_i = dream_imgs[i * net_params.batch_size: min((i + 1) * net_params.batch_size - 1, whole_size)]
            gen.optim.zero_grad()
            generated = gen(t_d_i)
            loss = t_criterion(generated, t_r_i)
            total_loss += loss.item()
            #print(f"    batch {i + 1} loss : {loss.item()}")
            loss.backward()
            gen.optim.step()
        print(f"epoch {epoch + 1} loss : {total_loss / blocks}")


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
    di = torch.transpose(torch.transpose(torch.tensor(di,dtype=torch.float32), 0, 2),1,2)
    ri = torch.transpose(torch.transpose(torch.tensor(ri, dtype=torch.float32), 0, 2),1,2)
    dream_img.append(di)
    real_img.append(ri)

r_datas = torch.stack(real_img,dim=0).to(device)
f_datas = torch.stack(dream_img, dim=0).to(device)
print("end loading data")
print("start training")
#train(train_net, r_datas, f_datas)

pretrain_for_generator(train_net, r_datas, f_datas, epoches=100)