import numpy as np
from bbox_tools import loc2bbox
import cupy as cp

class ProposalCreator:

    def __init__(self,
                 parent_model,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16
    ):
        self.parent_model=parent_model
        self.nms_thresh=nms_thresh
        self.n_train_pre_nms=n_train_pre_nms
        self.n_train_post_nms=n_train_post_nms
        self.n_test_pre_nms=n_test_pre_nms
        self.n_test_post_nms=n_test_post_nms
        self.min_size=min_size

    def __call__(self, loc, score,
                 anchor, img_size, scale=1.):
        
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms

        else:
            n_pre_nms= self.n_test_pre_nms
            n_post_nms=self.n_test_post_nms
        
        # Convert anchors into proposal via box transformations
        # loc is the predicted 
        # ref: http://www.telesens.co/2018/03/11/object-detection-and-classification-using-r-cnns/#ITEM-1455-4

        roi = loc2bbox(anchor, loc)

        # Clip predicted boxes to image
        roi[:, slice(0,4,2)] = np.clip(roi[:,slice(0,4,2)], 0, img_size[0])
        roi[:, slice(1,4,2)] = np.clip(roi[:,slice(1,4,2)], 0, img_size[1])

        min_size = self.min_size * scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = np.where((hs>=min_size)&(ws>=min_size))[0]
        roi = roi[keep,:]
        score = score[keep]

        keep = non_maximum_suppression(
            cp.ascontiguousarray(cp.asarray(roi)),
            thresh = self.nms_thresh
        )
        if n_post_nms>0:
            keep = keep[:n_post_nms]
        roi = roi[keep]
        return roi