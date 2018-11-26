import torch
from torch import nn
import torch.nn.functional as F


class FasterRCNN(nn.Module):
    def __init__(self, headNet, rpn, classifier):
        super().__init__()
        self.head= headNet 
        self.rpn = rpn
        self.classifier=classifier

    def forward(self, x):
        x=self.head(x)
        
