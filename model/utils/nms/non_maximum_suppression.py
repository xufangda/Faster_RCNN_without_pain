import numpy as np
import cupy as cp
import torch

try:
    from ._nms_gpu_post import _nms_gpu_post
except:
    import warnings
    warnings.warn('''
    the python code for non_maximum_suppression is about 2x slow
    It is strongly recommended to build cython code: 
    `cd model/utils/nms/; python3 build.py build_ext --inplace''')
    from ._nms_gpu_post_py import _nms_gpu_post
