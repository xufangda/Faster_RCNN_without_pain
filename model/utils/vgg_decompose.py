import torch
import torch.nn as nn
import torch.nn.functional as F

# get feature map
from torchvision.models.vgg import vgg16


def decom_vgg(use_drop=False):

    vggNet= vgg16(pretrained=True)

    features=vggNet.features
    features=nn.Sequential(*list(features.children())[:-1])
    # disable auto grad for first 9 layers
    for i in list(features.children())[:9]:
        i.requires_grad=False

    classifier=vggNet.classifier
    classifier = list(classifier)
    
    del classifier[6]
    if not use_drop:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)
    
    return features, classifier

FeatureNet, ClassifierNet = decom_vgg()
FeatureNet
ClassifierNet