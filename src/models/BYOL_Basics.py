import torchvision.models as models
import torch
from torch import nn


class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)


class ResNet18(torch.nn.Module):
    def __init__(self, mlp_hidden_size, projection_size):
        super(ResNet18, self).__init__()
        resnet = models.resnet18(pretrained=False)
        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projetion = MLPHead(
            in_channels=resnet.fc.in_features,
            mlp_hidden_size=mlp_hidden_size,
            projection_size=projection_size)

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projetion(h)
