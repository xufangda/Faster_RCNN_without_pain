import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RoIPooling2D(nn.Module):
    def __init__(self, outh=7, outw=7, spatial_scale=1.0 / 16.0):
        super().__init__()
        self.admax2d=nn.AdaptiveMaxPool2d((outh,outh))
        self.spatial_scale=spatial_scale

    def forward(self, features, rois, roi_indices):
        """
        Arg:
         - features (batch_size, channels, H, W)
         - rois (roi_number, 4)
         - roi_indices (roi_number,)
        Returen:
         - ret (N,7,7)
        """
        
        rois=rois*self.spatial_scale
        rois=rois.astype(np.int32)
        output=[]
        
        # roi_indices=roi_indices.to(dtype=torch.long)
        for i in range(rois.shape[0]):
            x1, y1, x2, y2=rois[i]
            ret=self.admax2d(features[roi_indices[i],:,y1:y2,x1:x2])
            output.append(ret)
        
        output=torch.stack(output,dim=0)
        return output

            