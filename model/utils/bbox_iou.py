import numpy as np

def bbox_iou(bbox_a, bbox_b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.

    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    inputs. Please note that both :obj:`bbox_a` and :obj:`bbox_b` need to be
    same type.
    The output is same type as the type of the inputs.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.

    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    """
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # bottom left
    bl = np.maximum(bbox_a[:,None,:2], bbox_b[:, :2])
    
    # top right
    tr = np.minimum(bbox_a[:,None,2:], bbox_b[:, 2:])

    area_i = np.prod(tr-bl, axis=2) * (bl < tr).all(axis=2)
    area_a = np.prod(bbox_a[:,2:] - bbox_a[:,:2], axis=1)
    area_b = np.prod(bbox_b[:,2:] - bbox_b[:,:2], axis=1)
    return area_i / (area_a[:,None] + area_b - area_i)