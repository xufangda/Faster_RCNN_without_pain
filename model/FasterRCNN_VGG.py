from .modules.extractor_network import decom_vgg
from .modules.region_proposal_network import RPN
from .modules.head_network import Head
from .modules.faster_rcnn import FasterRCNN

class FasterRCNN_VGG(FasterRCNN):
    def __init__(self, fg_n_class=20, feat_stride = 16.0):
        
        FeatureNet, Classifer = decom_vgg()

        R_Net=RPN(feat_stride=feat_stride)

        Head_Net= Head(n_class = fg_n_class +1,
                       pre_classifer = Classifer,
                       spatial_scale = 1.0/feat_stride)

        super().__init__(FeatureNet, R_Net, Head_Net)
        
    