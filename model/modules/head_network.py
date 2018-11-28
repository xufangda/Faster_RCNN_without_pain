import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.modules.extractor_network import ClassifierNet
from model.utils.roi_pool.roi_pooling_cupy import RoIPooling2D



class Head(nn.Module):
    """
    Args:
        - n_class: n_class, please note you should add one more class for background 
        e.g.: cat, dog should be 3 for [background, cat, dog]
        - use_drop: default False, decide whether use the dropout or not
    """
    def __init__(self, n_class, pre_classifer = ClassifierNet):
        
        super().__init__()
        self.n_class = n_class
        self.RoIPool=RoIPooling2D()
        self.pre_classifier = pre_classifer
        self.cls_loc = nn.Linear(4096, n_class*4)
        self.cls_score = nn.Linear(4096, n_class)
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.cls_score, 0, 0.01)

    def forward(self, featureMap, rois, roi_indices):
        """
        Args:
         - featureMap
         - rois
         - roi_indices
        """
        # Roi Pooling
        output=self.RoIPool(featureMap, rois, roi_indices)

        # flatten
        output = output.view(output.size(0),-1)
        
        # Fully connected layer
        feature=self.pre_classifier(output)
        final_loc=self.cls_loc(feature)
        final_score=self.cls_score(feature)

        return final_loc, final_score
    
def normal_init(m, mean, stddev, truncated=False):
    """
    weight initilizer: truncated normal and random normal
    """

    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean,stddev)
        m.bias.data.zero_()