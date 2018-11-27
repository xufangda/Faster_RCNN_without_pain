import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.anchor_generator import anchor_generator
import numpy as np
from model.utils.creator_tool import ProposalCreator

class RPN(nn.Module):
    def __init__(self,in_channel=512,mid_channel=512, feat_stride=16, n_anchor=9,proposal_creator_params=dict()):
        super().__init__()
        self.conv1=nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1)
        self.score=nn.Conv2d(mid_channel, 2*n_anchor, kernel_size=1, stride=1, padding=0)
        self.loc  =nn.Conv2d(mid_channel, 4*n_anchor, kernel_size=1, stride=1, padding=0)
        self.anchor_base= anchor_generator()
        self.feat_stride=feat_stride
        self.proposal_layer=ProposalCreator(self,**proposal_creator_params)
        normal_init(self.conv1, 0, 0.001)
        normal_init(self.score, 0, 0.001)
        normal_init(self.loc, 0, 0.001)

    def forward(self, featureMap, img_size, scale=1.):
        """
        Arg:
            - featureMap: (N, C, H, W)
        Return:
            - score: (N*H*W*9, 2)
            - loc: (N*H*W*9, 4)
        """
        
        N,C,H,W=featureMap.shape
        feature = self.conv1(featureMap)
        score = self.score(feature)
        loc = self.loc(feature)
        anchor= _enumberate_shifted_anchor(
            np.array(self.anchor_base),
            self.feat_stride, H, W
        )


        loc = loc.permute(0,2,3,1).contiguous().view(N, -1, 4)
        score= score.permute(0,2,3,1).contiguous().view(N,-1,2)
        score= F.softmax(score, dim=2)
        fg_score=score[:,:,1]


        rois=list()
        roi_indices=list()

        for i in range(N):
            roi = self.proposal_layer(
                loc[i].cpu().data.numpy(),
                fg_score[i].cpu().data.numpy(),
                anchor, img_size,
                scale=scale
            )
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = np.concatenate(rois,axis=0)
        roi_indices=np.concatenate(roi_indices, axis=0)

        return score, loc, rois, roi_indices, anchor


def _enumberate_shifted_anchor(anchor_base, feat_stride, height, width):
    """
    Arg:
     - anchor_base: (9, 4) array 
     - feat_stride: points distance
     - height: the height of the feature map
     - width : the width of the feature map
    Return:
     - anchors (height*width*9, 4)
    """
    shift_x=np.arange(0,width*feat_stride,feat_stride)
    shift_y=np.arange(0,height* feat_stride, feat_stride)

    shift_x, shift_y= np.meshgrid(shift_x,shift_y)
    
    shift_x=np.ravel(shift_x)
    shift_y=np.ravel(shift_y)
    
    shift = np.stack((shift_x,shift_y,shift_x,shift_y),axis=1)

    A=anchor_base.shape[0]
    K=shift.shape[0]

    anchor= anchor_base.reshape((1,A,4))
    shift = shift.reshape(1,K,4).transpose((1,0,2))

    ret_anchor=anchor+shift
    ret_anchor=ret_anchor.reshape(K*A, 4)
    return ret_anchor

def normal_init(m, mean, stddev, truncated=False):
    """
    weight initilizer: truncated normal and random normal
    """

    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean,stddev)
        m.bias.data.zero_()

