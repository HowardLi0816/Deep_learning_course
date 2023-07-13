import torch
import torch.nn as nn
import argparse
from torchsummary import summary


class CNN5(nn.Module):
    def __init__(self, img_size=224, input_channels=3, output_channels_1=32,
                 output_channels_2=64, output_channels_3=128, output_channels_4=256,
                 kernel_size=3, padding=1, pool_size=2, pool_stride=2, flatten=14,
                 relu_imp=False, drop_rate=0.1, linear_size=512, n_classes=29,
                 *args, **kwargs):
        super(CNN5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels_1, kernel_size, padding=padding),
            nn.ReLU(relu_imp),
            nn.MaxPool2d(pool_size, pool_stride)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(output_channels_1, output_channels_2, kernel_size, padding=padding),
            nn.ReLU(relu_imp),
            nn.MaxPool2d(pool_size, pool_stride)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(output_channels_2, output_channels_3, kernel_size, padding=padding),
            nn.ReLU(relu_imp),
            nn.MaxPool2d(pool_size, pool_stride)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(output_channels_3, output_channels_4, kernel_size, padding=padding),
            nn.ReLU(relu_imp),
            nn.MaxPool2d(pool_size, pool_stride)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(output_channels_4, output_channels_4, kernel_size, padding=padding),
            nn.ReLU(relu_imp),
            nn.MaxPool2d(pool_size, pool_stride)
        )
        self.layer6 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(output_channels_4 * flatten * flatten, linear_size),
            nn.ReLU(relu_imp),
            nn.Dropout(drop_rate)
        )
        self.classifier = nn.Linear(linear_size, n_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.classifier(x)
        return x

class CNN4(nn.Module):
    def __init__(self, img_size=112, input_channels=3, output_channels_1=32,
                 output_channels_2=64, output_channels_3=128, output_channels_4=256,
                 kernel_size=3, padding=1, pool_size=2, pool_stride=2, flatten=7,
                 relu_imp=False, drop_rate=0.1, linear_size=512, n_classes=29,
                 *args, **kwargs):
        super(CNN4, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels_1, kernel_size, padding=padding),
            nn.ReLU(relu_imp),
            nn.MaxPool2d(pool_size, pool_stride)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(output_channels_1, output_channels_2, kernel_size, padding=padding),
            nn.ReLU(relu_imp),
            nn.MaxPool2d(pool_size, pool_stride)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(output_channels_2, output_channels_3, kernel_size, padding=padding),
            nn.ReLU(relu_imp),
            nn.MaxPool2d(pool_size, pool_stride)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(output_channels_3, output_channels_4, kernel_size, padding=padding),
            nn.ReLU(relu_imp),
            nn.MaxPool2d(pool_size, pool_stride)
        )
        self.layer5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(output_channels_4 * flatten * flatten, linear_size),
            nn.ReLU(relu_imp),
            nn.Dropout(drop_rate)
        )
        self.classifier = nn.Linear(linear_size, n_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.classifier(x)
        return x

