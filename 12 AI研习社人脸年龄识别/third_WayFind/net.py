import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19_bn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        model = vgg19_bn(pretrained=True)
        # model = vgg19_bn(pretrained=False)
        weight = model.features[0].weight
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.features[0].weight = torch.nn.Parameter(weight[:, :1, :, :])
        self.features = nn.Sequential(*list(model.features.children())[:-4])
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 70),          
        )
    
    def forward(self, input):
        x = self.features(input)
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output
