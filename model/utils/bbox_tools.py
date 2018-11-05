import numpy as np


def loc2bbox(src_bbox, loc):
    """
    y1, x1, y2, x2 -> center x, center y , height, width

    Args:
    - src_bbox:
        np.array() size N x 4 
        (y_min, x_min, y_max, x_max)

    - loc:
    """

    if src_bbox==0:
        return np.zeros((0,4), dtype=loc.dtype)

    src_bbox=src_bbox.astype(src_bbox.dtype , copy=False)

    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width
    
    dy = loc[:,0::4] # step size 4
    dx = loc[:,1::4] # step size 4
    dh = loc[:,2::4] # step size 4
    dw = loc[:,3::4] # step size 4

    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
    h = np.exp(dh) * src_height[:, np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]

    dst_bbox = np.zeros(loc.shape, dtype = loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox


def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                         anchor_scales=[8,16,32]):
    """
    This function is used to generated 9 anchors.

    The size of 9 corresponding anchors are:

            width           height
    1       16x8/sqrt(0.5)  16x8xsqrt(0.5)
    2       16x8            16x8
    3       16x8/sqrt(2)    16x8xsqrt(2)
    4       16x16/sqrt(0.5) 16x16xsqrt(0.t)  
    5       ...
    .
    .
    9

    Args:

    - base_size (int): The width and the height of the reference window
    - ratios(list of floats): This is ratios of width to height fo the anchors
    - anchor_scales (list of numbers): This is areas of anchors. 
        These areas will be the product of the square of an element in 
        anchor_scales and the original area of the reference window

    Returns:
    
    - numpy.ndarray:
    - An array of shape (R, 4)

    """

    px = base_size / 2.
    py = base_size / 2.

    anchor_base=np.zeros((len(ratios)*len(anchor_scales), 4), dtype=np.float32)
    
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h/2
            anchor_base[index, 1] = px - w/2
            anchor_base[index, 2] = py + h/2
            anchor_base[index, 3] = px + w/2
    return anchor_base


if __name__=='__main__':
    ret = generate_anchor_base()
    print(ret)