import cupy as np

def loc2bbox(src_bbox, loc):
    """
    Args:
        - src_bbox: (R x 4) array, coordinate of downleft and upright points
        (downleft_x, downleft_y, upright_x, upright_y)

        - loc: (R x 4) array, scale parameters from network (tx, ty, tw, th)
    Return:
        - target_bbox: (R x 4) array,  coordinate of downleft and upright points
        (downleft_x, downleft_y, upright_x, upright_y)
    """

    src_w = src_bbox[:,2] - src_bbox[:,0]
    src_h = src_bbox[:,3] - src_bbox[:,1] 
    src_ctr_x = src_bbox[:,0] + 0.5 * src_w
    src_ctr_y = src_bbox[:,1] + 0.5 * src_h

    tx = loc[:,0::4]
    ty = loc[:,1::4]
    tw = loc[:,2::4]
    th = loc[:,3::4]

    tgt_w = np.exp(tw) * src_w[:,np.newaxis]
    tgt_h = np.exp(th) * src_h[:,np.newaxis]
    tgt_ctr_x = src_ctr_x[:,np.newaxis] + tx * src_w[:,np.newaxis]
    tgt_ctr_y = src_ctr_y[:,np.newaxis] + ty * src_h[:,np.newaxis]

    dst_bbox = np.zeros_like(loc,dtype=loc.dtype)
    dst_bbox[:,0::4] = tgt_ctr_x - 0.5 * tgt_w
    dst_bbox[:,1::4] = tgt_ctr_y - 0.5 * tgt_h
    dst_bbox[:,2::4] = tgt_ctr_x + 0.5 * tgt_w
    dst_bbox[:,3::4] = tgt_ctr_y + 0.5 * tgt_h

    return dst_bbox

 
