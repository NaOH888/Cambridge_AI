import math


from numpy import ndarray
import torch.nn as nn
import torch
import net_params



class AwakeNetGen(nn.Module):
    """
    by wzl
    Generator for AwakeNet
    """
    def __init__(self):
        super(AwakeNetGen, self).__init__()
        # in : [batch_size, 3, 512, 512]
        self.conv1 = nn.ConvTranspose2d(in_channels=net_params.input_image_depth,
                               out_channels=10,
                               kernel_size=(3,3),
                               stride=2,
                               padding=5)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=2)
        self.activ1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = nn.ConvTranspose2d(in_channels=10,
                               out_channels=10,
                               kernel_size=(3,3),
                               stride=2,
                               padding=5)
        self.pool2 = nn.MaxPool2d(kernel_size=5)
        self.activ2 = nn.LeakyReLU(negative_slope=0.2)
        self.conv3 = nn.ConvTranspose2d(in_channels=10,
                               out_channels=10,
                               kernel_size=(5, 5),
                               stride=2,
                               padding=4)
        self.pool3 = nn.AdaptiveAvgPool3d(output_size=(10,1024, 1024 ))
        self.activ3 = nn.LeakyReLU(negative_slope=0.2)
        # 现在应该是 [batch, 10, 1024, 1024], 然后第二维求和，变成[batch,1024, 1024]
        self.fc1 = nn.Linear(in_features=1024, out_features=2048,bias=True)
        self.activ4 = nn.LeakyReLU(negative_slope=0.2)
        # [batch,2048, 1024]
        self.fc2 = nn.Linear(in_features=2048, out_features=768, bias=True)
        self.activ5 = nn.Tanh()
        # then resize to [batch, 3, 512, 512]
        self.optim = torch.optim.Adam(self.parameters(), lr=net_params.learning_rate)


    def forward(self, x : torch.Tensor):
        if x.shape[1:] != (
                       net_params.input_image_depth,
                       net_params.input_image_size_x,
                       net_params.input_image_size_y):
            raise ValueError(fr"Input's size should be "
                             fr"({net_params.batch_size, 
                             net_params.input_image_depth, 
                             net_params.input_image_size_x,net_params.input_image_size_y})"
                             fr",but it is {x.shape}")
        bs = x.shape[0]
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.activ1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.activ2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.activ3(x)
        x = torch.sum(x, dim=1, keepdim=False)
        x = torch.reshape(x, shape=[bs,1024, 1024])
        x = self.fc1(x)
        x = self.activ4(x)
        x = self.fc2(x)
        x = (self.activ5(x) + 1) / 2 * 255
        x = torch.reshape(x, shape=[bs, 3, 512, 512])
        return x
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in')
                nn.init.constant_(m.bias, 0.1)


class AwakeNetDis(nn.Module):
    def __init__(self):
        super(AwakeNetDis, self).__init__()
        self.conv11 = nn.Conv2d(in_channels=net_params.input_image_depth,
                                out_channels=1,
                                kernel_size=(5,5),
                                stride=2,
                                padding=4)
        self.conv12 = nn.Conv2d(in_channels=net_params.input_image_depth,
                                out_channels=1,
                                kernel_size=(5, 5),
                                stride=2,
                                padding=4)
        self.pool1 = nn.AdaptiveAvgPool3d(output_size=(1, 1024, 1024))
        self.activ1 = nn.LeakyReLU(negative_slope=0.2)
        # reshape to (batch, 1024, 1024) and cat together to (batch, 2048, 1024)
        self.fc1 = nn.Linear(in_features=1024, out_features=1024,bias=True)
        self.activ2 = nn.LeakyReLU(negative_slope=0.2)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024, bias=True)
        self.activ3 = nn.LeakyReLU(negative_slope=0.2)
        # flatten
        self.fc3 = nn.Linear(in_features=1024 * 2048, out_features=512, bias=True)
        self.activ4 = nn.LeakyReLU(negative_slope=0.2)
        self.fc4 = nn.Linear(in_features=512, out_features=1, bias=True)
        self.activ5 = nn.Sigmoid()


        self.optim = torch.optim.Adam(self.parameters(), lr=net_params.learning_rate)

    def forward(self, origin : torch.Tensor, dreamt : torch.Tensor):
        if origin.shape[1:] != (
                            net_params.input_image_depth,
                            net_params.input_image_size_x,
                            net_params.input_image_size_y):
            raise ValueError(fr"Input's size should be "
                             fr"{net_params.batch_size, 
                             net_params.input_image_depth,  
                             net_params.input_image_size_x,
                             net_params.input_image_size_y}"
                             fr",but it is {origin.shape}")
        if origin.shape != dreamt.shape:
            raise ValueError(fr"Origin's size should be same as Dreamt Size")
        bs = origin.shape[0]

        origin = self.conv11(origin)
        dreamt = self.conv12(dreamt)

        origin = self.pool1(origin)
        dreamt = self.pool1(dreamt)

        origin = self.activ1(origin)
        dreamt = self.activ1(dreamt)

        origin = torch.reshape(origin, shape=[bs, 1024, 1024])
        dreamt = torch.reshape(dreamt, shape=[bs, 1024, 1024])

        origin = torch.cat((origin, dreamt), dim=1)

        origin = self.fc1(origin)
        origin = self.activ2(origin)
        origin = self.fc2(origin)
        origin = self.activ3(origin)
        origin = torch.flatten(origin, start_dim=1, end_dim=-1)
        origin = self.fc3(origin)
        origin = self.activ4(origin)
        origin = self.fc4(origin)
        origin = self.activ5(origin)
        # out : [batch_size, 1]
        return origin
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in')
                nn.init.constant_(m.bias, 0.1)
class AwakeNet(nn.Module):
    def __init__(self, n_dis : int):
        super(AwakeNet, self).__init__()
        if n_dis < 1:
            raise ValueError(fr"n_dis should be a positive integer")
        self.generator = AwakeNetGen()
        self.discriminators = nn.Sequential()
        for i in range(n_dis):
            self.discriminators.add_module(str(i), AwakeNetDis())
    def forward(self, x : torch.Tensor, dream : torch.Tensor):

        # x & dream : [batch, depth, size_X, size_Y]

        if x.shape[1:] != (
                       net_params.input_image_depth,
                       net_params.input_image_size_x,
                       net_params.input_image_size_y):
            raise ValueError(fr"Input's size should be "
                             fr"{net_params.batch_size, 
                             net_params.input_image_depth,  
                             net_params.input_image_size_x, 
                             net_params.input_image_size_y}"
                             fr",but it is {x.shape}")
        if x.shape != dream.shape:
            raise ValueError(fr"The shape of raw image and dreamt image should be the same")


        generated_dream = self.generator(x)

        discriminate_real_score = []
        discriminate_fake_score_usedByDiscriminatorTrain = []
        discriminate_fake_score_usedByGeneratorTrain = []

        for dis in self.discriminators:
            discriminate_real_score.append(dis(x, dream))
            discriminate_fake_score_usedByDiscriminatorTrain.append(dis(generated_dream.detach(), dream))
            discriminate_fake_score_usedByGeneratorTrain.append(dis(generated_dream, dream))
        return (torch.stack(discriminate_real_score,dim=0),
                torch.stack(discriminate_fake_score_usedByDiscriminatorTrain,dim=0),
                torch.stack(discriminate_fake_score_usedByGeneratorTrain,dim=0))
    def initial(self):
        self.generator.initialize_weights()
        for dis in self.discriminators:
            dis.initialize_weights()

