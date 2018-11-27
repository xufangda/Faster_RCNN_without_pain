import numpy as np
import cupy as cp
import torch
import torch.nn.functional as F
from model.utils.bbox_tool import loc2bbox
from model.utils.nms import non_maximum_suppression

def suppression(rois,
                roi_indices,
                final_loc,
                final_score, 
                n_class,
                img_size,
                img_number,
                high_thresh=True):
    """
    rois: numpy.ndarray(N, 4)
    final_loc: torch.tensor(N,4*n_class)
    final_score: torch.tensor(N, n_class)
    n_class: int
    img_size: numpy.ndarray(2,)
    """

    if high_thresh is True: 
        # default on
        nms_thresh = 0.3
        score_thresh = 0.7
    else: 
        # visualization mode high score threshold
        nms_thresh = 0.3
        score_thresh = 0.05
    
    
    final_loc = final_loc.view(-1, n_class, 4)
    final_loc = final_loc.cpu().data.numpy()
    rois = np.repeat(rois[:,np.newaxis,:], n_class, axis=1)
    
    final_bbox = loc2bbox(rois.reshape(-1,4), final_loc.reshape(-1,4))
    final_bbox[:,slice(0,4,2)] = np.clip(final_bbox[:,slice(0,4,2)],0,img_size[1])
    final_bbox[:,slice(1,4,2)] = np.clip(final_bbox[:,slice(1,4,2)],0,img_size[0])
    final_bbox = final_bbox.reshape(-1, n_class, 4)

    final_prob = F.softmax(final_score, dim = 1)
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


