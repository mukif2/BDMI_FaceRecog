from __future__ import print_function
import os
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from .models import *
import torch
import numpy as np
from torch.nn import DataParallel
import sys
def arcface(face):
    '''
    输入3x112x112的PIL Image，输出512D feature
    '''
    normalize = T.Normalize(mean=[0.5], std=[0.5])
    transforms = T.Compose([
        T.ToTensor(),
        normalize
    ])
    face = face.convert('L')
    face = transforms(face)
    face = face.float()

    device = torch.device("cuda")
    model = resnet_face18(use_se=True)
    model.to(device)
    model = DataParallel(model)
    model.load_state_dict(torch.load('arcface/checkpoints/resnet18_49.pth2'))
    with torch.no_grad():

        model.eval()
        face = torch.unsqueeze(face,0)
        face = face.to(device)
        feature = model(face)
        feature = F.normalize(feature,dim=1)
        return feature