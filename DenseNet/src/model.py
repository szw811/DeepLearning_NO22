import torch
import torch.nn as nn
import torchvision
import torchvision.models as models


class ModelDenseNet(nn.Module):

    def __init__(self):
        super(ModelDenseNet, self).__init__()
        model = models.densenet121(pretrained=True)
        self.feature_layer = model.features

        self.fc1 = nn.Sequential(
            nn.Linear(65536, 2),
            nn.BatchNorm1d(2),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(65536, 2),
            nn.BatchNorm1d(2),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(65536, 2),
            nn.BatchNorm1d(2),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(65536, 3),
            nn.BatchNorm1d(3),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.feature_layer(x)
        x = x.view(x.size(0), -1)

        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        x4 = self.fc4(x)
        return x1, x2, x3, x4




def main():
    tmp = torch.randn(2, 3, 32, 32)
    net = ModelDenseNet()
    print(net.feature_layer)
    # out, _ = net(tmp)
    # print(out.shape)


if __name__ == '__main__':
    main()


