import numpy as np
from utils.bbox_tools import generate_anchor_base 

class RegionProposalNetwork(nn.Module):

    def __init__(
        self, in_channels = 512, mid_channels = 512, ratios=[0.5, 1, 2],
        anchor_scales=[8, 16 ,32], feat_stride=16,
        proposal_creator_params=dict(),
    ):
        super().__init__()
        self.anchor_base = generate_anchor_base(
            anchor_scales = anchor_scales, ratios = ratios
        )
        self.feat_stride = feat_stride
        self.proposal_layer = 