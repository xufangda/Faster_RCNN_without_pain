from torch import nn
from torchvision.models import vgg16
from utils.config import opt

def decom_vgg16():
    model = vgg16(not opt.load_path)

    # select 29 layers in the model, because the last layer is a maxpool layer
    # 16x downsample
    features=list(model.features)[:30] 


    classifier = model.classifier
    del classifier[6]
    if not opt.use_drop:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    # freeze top4 conv layer (0,2,5,7)
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier

class FasterRCNNVGG16(FasterRCNN):

    feat_stride = 16 # downsample 16x for output of conv5 in vgg16

    def __init__(self, 
                 n_fg_class=20,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                ):
        
        extractor, classifier = decom_vgg16()

        rpn = 

        super().__init__(
            extractor,
            rpn,
            head,
        )