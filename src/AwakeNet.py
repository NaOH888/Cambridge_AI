import torch
import torch.nn as nn
from numpy import ndarray

import net_params

class AwakeNetGen(nn.Module):
    """
    by wzl
    Generator for AwakeNet
    """
    def __init__(self):
        super(AwakeNetGen, self).__init__()
        # in : [batch_size, 512, 512,
        self.conv1 = nn.Conv2d(in_channels=net_params.input_image_depth,
                               out_channels=10,
                               kernel_size=(5,5),
                               stride=2,
                               padding=4)
        self.pool1 = nn.AdaptiveAvgPool3d(output_size=(1024,1024,10))
        self.activ1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = nn.Conv2d(in_channels=10,
                               kernel_size=(5,5),
                               stride=2,
                               padding=4)
        self.pool2 = nn.AdaptiveAvgPool3d(output_size=(1024, 1024,10))
        self.activ2 = nn.LeakyReLU(negative_slope=0.2)
        self.conv3 = nn.Conv2d(in_channels=10,
                               kernel_size=(5, 5),
                               stride=2,
                               padding=4)
        self.pool3 = nn.AdaptiveAvgPool3d(output_size=(1024, 1024, 10))
        self.activ3 = nn.LeakyReLU(negative_slope=0.2)
        # 现在应该是 [batch, 1024, 1024, 10], 然后最后一维求和，变成[batch,2048, 5120]
        self.fc1 = nn.Linear(in_features=5120, out_features=1024,bias=True)
        self.activ4 = nn.LeakyReLU(negative_slope=0.2)
        # [batch,2048, 1024]
        self.fc2 = nn.Linear(in_features=1024, out_features=384, bias=True)
        self.activ5 = nn.Tanh()
        # then resize to [batch, 512, 512, 3]


    def forward(self, x : ndarray):
        if x.shape != [net_params.batch_size, net_params.input_image_depth, net_params.input_image_size]:
            raise ValueError(fr"Input's size should be "
                             fr"{net_params.batch_size, 
                             net_params.input_image_depth, 
                             net_params.input_image_size}"
                             fr",but it is {x.shape}")
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.activ1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.activ2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.activ3(x)
        x = torch.sum(x, dim=3, keepdim=False)
        x = torch.reshape(x, shape=[net_params.batch_size,2048, 5120])
        x = self.fc1(x)
        x = self.activ4(x)
        x = self.fc2(x)
        x = self.activ5(x)
        x = torch.reshape(x, shape=[net_params.batch_size, 512, 512, 3])
        return x

class AwakeNetDis(nn.Module):
    def __init__(self):
        super(AwakeNetDis, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=net_params.input_image_depth,
                               out_channels=1,
                               kernel_size=(5,5),
                               stride=2,
                               padding=4)
        self.pool1 = nn.AdaptiveAvgPool3d(output_size=(1024, 1024, 1))
        self.activ1 = nn.LeakyReLU(negative_slope=0.2)
        # reshape to (batch, 1024, 1024)
        self.fc1 = nn.Linear(in_features=1024, out_features=1024,bias=True)
        self.activ2 = nn.LeakyReLU(negative_slope=0.2)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024, bias=True)
        self.activ3 = nn.LeakyReLU(negative_slope=0.2)
        # squeeze
        self.fc3 = nn.Linear(in_features=1024 * 1024, out_features=512, bias=True)
        self.activ4 = nn.LeakyReLU(negative_slope=0.2)
        self.fc4 = nn.Linear(in_features=512, out_features=1, bias=True)
        self.activ5 = nn.Sigmoid()
    def forward(self, x : ndarray):
        if x.shape != [net_params.batch_size, net_params.input_image_depth, net_params.input_image_size]:
            raise ValueError(fr"Input's size should be "
                             fr"{net_params.batch_size, 
                             net_params.input_image_depth, 
                             net_params.input_image_size}"
                             fr",but it is {x.shape}")
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.activ1(x)
        x = torch.reshape(x, shape=[net_params.batch_size, 1024,1024])
        x = self.fc1(x)
        x = self.activ2(x)
        x = self.fc2(x)
        x = self.activ3(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.fc3(x)
        x = self.activ4(x)
        x = self.fc4(x)
        x = self.activ5(x)
        # out : [batch_size, 1]
        return x
