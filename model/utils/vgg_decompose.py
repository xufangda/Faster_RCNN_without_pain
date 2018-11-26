import torch
import torch.nn as nn
import torch.nn.functional as F

# get feature map
from torchvision.models.vgg import vgg13
vggNet= vgg13(pretrained=True)

features=vggNet.features
features=nn.Sequential(*list(features.children())[:-1])
classifier=vggNet.classifier



# disable auto grad for first 9 layers
for i in list(features.children())[:9]:
    i.requires_grad=False
