import numpy as np
from skimage import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torchvision.models import resnet50 

import numpy as np
from skimage import io
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T

from torchvision.models.vgg import vgg13
vggNet=vgg13(pretrained=True)

img=io.imread('demo_4.jpg')

plt.figure()
plt.imshow(img)
trsfm=T.ToTensor()
x=trsfm(img).unsqueeze(0)

print(device)

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

dtype=torch.float
vNet=vggNet.features
# feature=model34.to(device)
x=trsfm(img).unsqueeze(0)
x=x.to(dtype=dtype,device=device)
# # vggNet=vggNet.cuda()
vNet.cuda(device=device)

y=vNet(x)

# import torch.nn as nn
# model=nn.Sequential(nn.Conv2d(512,1,kernel_size=(1, 1), stride=(1, 1), padding=0))
# model=model.to(device)
# dtype=torch.float
# device=torch.device('cuda')
# x=x.to(dtype=dtype,device=device)