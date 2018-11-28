import cupy as cp
from model.utils.nms import non_maximum_suppression
from model.utils.loc2bbox_gpu import loc2bbox

from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

class ProposalCreator:
    def __init__(self, 
                 parent_model,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=1000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300, # due to the limited memory, n_test_post_nms is reduced to 50
                 min_size=16
                 ):
        self.parent_model=parent_model
        self.nms_thresh=nms_thresh
        self.n_train_pre_nms=n_train_pre_nms
        self.n_train_post_nms=n_train_post_nms
        self.n_test_pre_nms=n_test_pre_nms
        self.n_test_post_nms=n_test_post_nms
        self.min_size=min_size

    def __call__(self, loc, fg_score, anchor, img_size, scale=1.):
        """
        Arg:
         - loc: (N,4)
         - fg_score: (N,)
         - anchor: (9, 4)
         - img_size: (2)
        """


        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        loc = cp.fromDlpack(to_dlpack(loc))
        fg_score = cp.fromDlpack(to_dlpack(fg_score))
        anchor = cp.asarray(anchor)
        roi=loc2bbox(anchor, loc)
        
        # clip 
        roi[:,slice(0,4,2)]=cp.clip(roi[:,slice(0,4,2)], 0, img_size[1])
        roi[:,slice(1,4,2)]=cp.clip(roi[:,slice(1,4,2)], 0, img_size[0])

        # remove small box less than threshold
        min_size=self.min_size * scale
        hs = roi[:,3]-roi[:,1]
        ws = roi[:,2]-roi[:,0]
        keep=cp.where((hs>min_size) & (ws>min_size))[0]
        roi=roi[keep,:]
        fg_score=fg_score[keep]

        # sort the score
        order= cp.argsort(fg_score.ravel())[::-1]
        if n_pre_nms>0:
            order= order[0:n_pre_nms]
        roi=roi[order,:]
        
        keep = non_maximum_suppression(cp.ascontiguousarray(cp.asarray(roi)), thresh = self.nms_thresh)

        if n_post_nms>0:
            keep = keep[:n_post_nms]
        roi=roi[keep]
        return roi