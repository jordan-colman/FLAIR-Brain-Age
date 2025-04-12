import torch
import torch.nn as nn

## define core 3D convolutional unit
class Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, bias=True):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels, eps=0.001, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

## define initial stem of architecture
class Stem(nn.Module):
    def __init__(self, in_channels):
        super(Stem, self).__init__()
        self.features = nn.Sequential(
            Conv3d(in_channels, 32, 3, stride=2, padding=1), # 149 x 149 x 32
            Conv3d(32, 32, 3, stride=1, padding=1), # 147 x 147 x 32
            Conv3d(32, 64, 3, stride=1, padding=1), # 147 x 147 x 64
        )
        
        self.branch_a = Conv3d(64, 96, 3, stride=2, padding=1)
        self.branch_b = nn.MaxPool3d(3, stride=2, padding=1)
        
        self.branch_0 = nn.Sequential(
            Conv3d(160, 64, 1, stride=1, padding=0),
            Conv3d(64, 96, 3, stride=1, padding=1)
        )
        self.branch_1 = nn.Sequential(
            Conv3d(160, 64, 1, stride=1, padding=0),
            Conv3d(64, 64, (1,1,7), stride=1, padding=(0,0,3)),
            Conv3d(64, 64, (1,7,1), stride=1, padding=(0,3,0)),
            Conv3d(64, 64, (7,1,1), stride=1, padding=(3,0,0)),
            Conv3d(64, 96, 3, stride=1, padding=1)
        )
        self.branch_2 = Conv3d(192, 192, 3, stride=2, padding=1)
        self.branch_3 = nn.MaxPool3d(3, stride=2, padding=1)
    def forward(self, x):
        x = self.features(x)
        xa = self.branch_a(x)
        xb = self.branch_b(x)
        x = torch.cat((xa,xb),dim=1)
        
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x = torch.cat((x0,x1),dim=1)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat((x2, x3), dim=1)

## define seperate blocks 

class Reduction_A(nn.Module):
    # 35 -> 17
    def __init__(self, in_channels, k, l, m, n):
        super(Reduction_A, self).__init__()
        self.branch_0 = Conv3d(in_channels, n, 3, stride=2, padding=1)
        self.branch_1 = nn.Sequential(
            Conv3d(in_channels, k, 1, stride=1, padding=0),
            Conv3d(k, l, 3, stride=1, padding=1),
            Conv3d(l, m, 3, stride=2, padding=1),
        )
        self.branch_2 = nn.MaxPool3d(3, stride=2, padding=1)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        return torch.cat((x0, x1, x2), dim=1) # 17 x 17 x 1024

class Inception_ResNet_A(nn.Module):
    def __init__(self, in_channels, scale=1.0):
        super(Inception_ResNet_A, self).__init__()
        self.scale = scale
        self.branch_0 = Conv3d(in_channels, 32, 1, stride=1, padding=0)
        self.branch_1 = nn.Sequential(
            Conv3d(in_channels, 32, 1, stride=1, padding=0),
            Conv3d(32, 32, 3, stride=1, padding=1)
        )
        self.branch_2 = nn.Sequential(
            Conv3d(in_channels, 32, 1, stride=1, padding=0),
            Conv3d(32, 48, 3, stride=1, padding=1),
            Conv3d(48, 64, 3, stride=1, padding=1)
        )
        self.conv = nn.Conv3d(128, 384, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x_res = torch.cat((x0, x1, x2), dim=1)
        x_res = self.conv(x_res)
        return self.relu(x + self.scale * x_res)


class Inception_ResNet_B(nn.Module):
    def __init__(self, in_channels, scale=1.0):
        super(Inception_ResNet_B, self).__init__()
        self.scale = scale
        self.branch_0 = Conv3d(in_channels, 192, 1, stride=1, padding=0)
        self.branch_1 = nn.Sequential(
            Conv3d(in_channels, 128, 1, stride=1, padding=0),
            Conv3d(128, 160, (1, 1, 7), stride=1, padding=(0, 0, 3)),
            Conv3d(160, 160, (1, 7, 1), stride=1, padding=(0, 3, 0)),
            Conv3d(160, 192, (7, 1, 1), stride=1, padding=(3, 0, 0))
        )
        self.conv = nn.Conv3d(384, 1152, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)
        return self.relu(x + self.scale * x_res)


class Reduciton_B(nn.Module):
    def __init__(self, in_channels):
        super(Reduciton_B, self).__init__()
        self.branch_0 = nn.Sequential(
            Conv3d(in_channels, 256, 1, stride=1, padding=0),
            Conv3d(256, 384, 3, stride=2, padding=1)
        )
        self.branch_1 = nn.Sequential(
            Conv3d(in_channels, 256, 1, stride=1, padding=0),
            Conv3d(256, 288, 3, stride=2, padding=1),
        )
        self.branch_2 = nn.Sequential(
            Conv3d(in_channels, 256, 1, stride=1, padding=0),
            Conv3d(256, 288, 3, stride=1, padding=1),
            Conv3d(288, 320, 3, stride=2, padding=1)
        )
        self.branch_3 = nn.MaxPool3d(3, stride=2, padding=1)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)


class Inception_ResNet_C(nn.Module):
    def __init__(self, in_channels, scale=1.0, activation=True):
        super(Inception_ResNet_C, self).__init__()
        self.scale = scale
        self.activation = activation
        self.branch_0 = Conv3d(in_channels, 192, 1, stride=1, padding=0)
        self.branch_1 = nn.Sequential(
            Conv3d(in_channels, 192, 1, stride=1, padding=0),
            Conv3d(192, 224, (1, 1, 3), stride=1, padding=(0, 0, 1)),
            Conv3d(224, 224, (1, 3, 1), stride=1, padding=(0, 1, 0)),
            Conv3d(224, 256, (3, 1, 1), stride=1, padding=(1, 0, 0))
        )
        self.conv = nn.Conv3d(448, 2144, 1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x_res = torch.cat((x0, x1), dim=1)
        x_res = self.conv(x_res)
        if self.activation:
            return self.relu(x + self.scale * x_res)
        return x + self.scale * x_res

## define overal architecture using above blocks

class Inception_ResNetv2(nn.Module):
    def __init__(self, in_channels=1, dropout=0.2, classes=1, k=256, l=256, m=384, n=384, lin_regression=True):
        super(Inception_ResNetv2, self).__init__()
        blocks = []
        blocks.append(Stem(in_channels))
        for i in range(5):
            blocks.append(Inception_ResNet_A(384, 0.2))
        blocks.append(Reduction_A(384, k, l, m, n))
        for i in range(10):
            blocks.append(Inception_ResNet_B(1152, 0.2))
        blocks.append(Reduciton_B(1152))
        for i in range(4):
            blocks.append(Inception_ResNet_C(2144, 0.2))
        blocks.append(Inception_ResNet_C(2144, activation=False))
        self.features = nn.Sequential(*blocks)
        self.conv = Conv3d(2144, 1536, 1, stride=1, padding=0)
        self.global_average_pooling = nn.Sequential(
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Dropout(p=dropout)
        )
        
        if lin_regression==True:
        
          self.linear = nn.Sequential(
            nn.Linear(1536, 1),
            nn.Linear(1,1)
          )
        else:
          self.linear = nn.Linear(1536, classes)

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.global_average_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
