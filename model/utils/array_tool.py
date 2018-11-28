"""
tools to convert specified type
"""
import torch 
import numpy as np


def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()


def totensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data).to(dtype=torch.float)
    if isinstance(data, torch.Tensor):
        tensor = data.detach()
    if cuda:
        tensor = tensor.cuda()
    return tensor