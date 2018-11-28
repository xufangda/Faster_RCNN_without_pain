import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cupy as cp
from model.utils.loc2bbox import loc2bbox
from model.utils.nms import non_maximum_suppression

class FasterRCNN(nn.Module):
    """Base class for Faster R-CNN.

    This is a base class for Faster R-CNN links supporting object detection
    API [#]_. The following three stages constitute Faster R-CNN.

    1. **Feature extraction**: Images are taken and their \
        feature maps are calculated.
    2. **Region Proposal Networks**: Given the feature maps calculated in \
        the previous stage, produce set of RoIs around objects.
    3. **Localization and Classification Heads**: Using feature maps that \
        belong to the proposed RoIs, classify the categories of the objects \
        in the RoIs and improve localizations.
    """
    def __init__(self, extractor, rpn, head,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
                ):
                      
        self.extractor = extractor
        self.rpn = rpn
        self.head = head

    @property
    def n_class(self):
        # Total number of classes including the background.
        return self.head.n_class

    def forward(self, x, scale=1.0):
        """
        Args:
         - x: [N, C, H, W]
         - scale (float): Amount of scaling applied to the raw image
            during preprocessing.
        
        Returns:
         - final_locs: [N, classes * 4]
         - final_scores: [N, classes * 1]
         - rois: [K, 4]
         - roi_indices: [K,]
        """
        img_size = x.shape[2:]

        featureMap = self.extractor(x)

        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.rpn(featureMap, img_size, scale)
        
        final_locs, final_scores = self.head(
            featureMap, rois, roi_indices
        )
        return final_locs, final_scores, rois, roi_indices
    
    def predict(self, imgs):
        
        N, C, H, W = imgs.shape   
        img_size = (H, W)
        img_number = N
        
        final_locs, final_scores, rois, roi_indices = self.forward(imgs)

        bboxes, labels, scores = self._suppression(rois, roi_indices,
                                                   final_locs, final_scores,
                                                   img_size,
                                                   img_number)
        return bboxes, labels, scores

def get_optimizer(self):
        """
        return optimizer, It could be overwriten if you want to specify 
        special optimizer
        """
        
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': 1e-3 * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': 1e-3, 'weight_decay': 0.1}]

        self.optimizer = torch.optim.SGD(params, momentum=0.9)
        return self.optimizer

    def _suppression(self, rois, roi_indices,
                     final_locs, final_scores,
                     img_size,
                     img_number,
                     high_thresh=True):
        """
        rois: numpy.ndarray(N, 4)
        roi_indices: numpy.ndarray(N)
        final_locs: torch.tensor(M,4*n_class)
        final_scores: torch.tensor(M, n_class)
        img_size: numpy.ndarray(2,)
        img_number: int
        
        """

        if high_thresh is True: 
            # default on
            nms_thresh = 0.3
            score_thresh = 0.7
        else: 
            # visualization mode high score threshold
            nms_thresh = 0.3
            score_thresh = 0.05
        
        n_class = self.n_class
        final_locs = final_locs.view(-1, n_class, 4)
        final_locs = final_locs.cpu().data.numpy()
        rois = np.repeat(rois[:,np.newaxis,:], n_class, axis=1)
        
        final_bbox = loc2bbox(rois.reshape(-1,4), final_locs.reshape(-1,4))
        final_bbox[:,slice(0,4,2)] = np.clip(final_bbox[:,slice(0,4,2)],0,img_size[1])
        final_bbox[:,slice(1,4,2)] = np.clip(final_bbox[:,slice(1,4,2)],0,img_size[0])
        final_bbox = final_bbox.reshape(-1, n_class, 4)

        final_prob = F.softmax(final_scores, dim = 1)
        final_prob = final_prob.cpu().data.numpy()

        bboxes = list()
        labels = list()
        scores = list()

        # select each single image 
        for cnt in range(img_number):
            cnt_mask=np.where(roi_indices==cnt)
            bbox = list()
            label = list()
            score = list()

            # skip cls_id = 0 for it is background class
            for i in range(1, n_class):
                i_bbox = final_bbox[cnt_mask][:,i,:]
                i_prob = final_prob[cnt_mask][:,i]
                
                mask = i_prob > score_thresh
                # mask bbox and prob
                i_bbox = i_bbox[mask]
                i_prob = i_prob[mask]
                keep = non_maximum_suppression(
                    cp.array(i_bbox), nms_thresh, i_prob
                )
                keep = cp.asnumpy(keep)
                bbox.append(i_bbox[keep])
                label.append((i-1) * np.ones((len(keep),)))
                score.append(i_prob[keep])
            bbox = np.concatenate(bbox, axis=0).astype(np.float32)
            label = np.concatenate(label, axis=0).astype(np.int32)
            score = np.concatenate(score, axis=0).astype(np.float32)
            
            # final bbox, label and score for a single image
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        return bboxes, labels, scores
