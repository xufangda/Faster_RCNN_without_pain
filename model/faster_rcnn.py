import torch
from torch import nn
import torch.nn.functional as F
from model.utils.vgg_decompose import FeatureNet, ClassifierNet
from model.utils.rpn import RPN
from model.utils.vgg_roi_head import VGG_RoI_Head
from model.utils.final_suppression import suppression

class FasterRCNN(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.featureNet = FeatureNet
        self.rpn = RPN()
        self.head= VGG_RoI_Head(n_class=10)
        self.n_class = n_class

    def forward(self, imgs):
        """
        img: (N, C ,H, W)
        """
        N, C, H, W = imgs.shape 
        featureMap=self.featureNet(imgs)
        img_size=[H, W]
        score, loc, rois, roi_indices, anchor = self.rpn(featureMap, img_size)
        final_loc, final_score=self.head(featureMap, rois , roi_indices)
        
        return final_loc, final_score, rois, roi_indices
    
    def predict(self,imgs):
        N, C, H, W = imgs.shape
        img_size = [H, W]
        final_loc, final_score, rois, roi_indices = self.forward(imgs)
        bboxes, labels, scores=suppression(rois, roi_indices, final_loc, final_score,
                                           self.n_class, img_size, N)
                                           
        return bboxes, labels, scores