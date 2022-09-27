import torch
import torch.nn as nn
from torch.nn import functional as F

from vstools.img_tools.img_proc import get_img_from_img_p

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class ColorBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 48, 11, 4),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.MaxPool2d(3, 2)
        )

        self.top_conv2 = nn.Sequential(
            LambdaLayer(lambda x : x[:, :24, :, :]),
            nn.Conv2d(24, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 96, 3, 1, 1),
            nn.ReLU()
        )

        self.bot_conv2 = nn.Sequential(
            LambdaLayer(lambda x : x[:, 24:, :, :]),
            nn.Conv2d(24, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 96, 3, 1, 1),
            nn.ReLU()
        )

        self.top_conv3 = nn.Sequential(
            LambdaLayer(lambda x : x[:, :96, :, :]),
            nn.Conv2d(96, 96, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(96, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.bot_conv3 = nn.Sequential(
            LambdaLayer(lambda x : x[:, 96:, :, :]),
            nn.Conv2d(96, 96, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(96, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )


    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)

        x_top = self.top_conv2(x)
        x_bot = self.bot_conv2(x)
        # print(x_top.shape, x_bot.shape)

        x = torch.cat((x_top, x_bot), 1)
        # print(x.shape)

        x_top = self.top_conv3(x)
        x_bot = self.bot_conv3(x)
        # print(x_top.shape, x_bot.shape)

        return x_top, x_bot

class ColorNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.top = ColorBlock()
        self.bot = ColorBlock()

        self.fc_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6400, 4096),
            nn.Dropout(0.6),
            nn.ReLU(),
        )

        self.fc_2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.Dropout(0.6),
            nn.ReLU(),
        )

        self.fc_3 = nn.Sequential(
            nn.Linear(4096, num_classes),
            nn.Softmax(1),
        )

    def forward(self, x):
        x_top = self.top(x)
        x_bot = self.top(x)

        x = torch.cat((x_top[0], x_top[1], x_bot[0], x_bot[1]), 1)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)

        # print(x.shape)
        return x

