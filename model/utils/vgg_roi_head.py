import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.utils.vgg_decompose import classifier


class VGG_RoI_Head(nn.Module):
    def __init__(self,n_class,VGG_classifer=classifier, use_drop=False):
        super().__init__()
        
        classify = list(classifier)
        del classify[6]
        if not use_drop:
            del classify[5]
            del classify[2]
        classify = nn.Sequential(*classify)
        
        self.pre_classifier=classify
        self.cls_loc = nn.Linear(4096, n_class*4)
        self.cls_score = nn.Linear(4096, n_class)


    def forward(self,x, rois, roi_indices):
        feature=self.pre_classifier(x)
        final_loc=self.cls_loc(feature)
        final_score=self.cls_score(feature)

        return final_loc, final_score