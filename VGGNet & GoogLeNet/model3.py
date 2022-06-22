import torch
from torch import nn
from d2l import torch as d2l


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)


def vgg(conv_arch):
    conv_blks = []
    in_channels = 3
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(32768, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 128))



# VGG
class MyModle_VGG(nn.Module):

    def __init__(self):
        super(MyModle_VGG, self).__init__()

        conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

        self.feature_layer = nn.Sequential(
            vgg(conv_arch)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(128, 2),
            nn.BatchNorm1d(2),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 5),
            nn.BatchNorm1d(5),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128, 2),
            nn.BatchNorm1d(2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.feature_layer(x)
        x = x.view(x.size(0), -1)

        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        return x1, x2, x3

def main():
    tmp = torch.randn(2, 3, 32, 32)
    net = MyModle_VGG()
    # out, _ = net(tmp)
    # print(out.shape)


if __name__ == '__main__':
    main()
