from typing import Union

import torch
from torch import nn, Tensor
from torch.nn.functional import normalize

from auxiliary.settings import USE_CONFIDENCE_WEIGHTED_POOLING, DEVICE
from classes.fc4.repvit.Repvit import repvit_m0_6
from classes.fc4.repvit.utils import replace_batchnorm
from timm.models import create_model
import numpy as np
import timm
from torch import nn, einsum
from einops import rearrange, repeat
import math
from inspect import isfunction


    

class FC4(torch.nn.Module):

    def __init__(self, img: Tensor=None):
        super().__init__()

        repvit = repvit_m0_6(False, 1000, False)

        self.backbone1 = repvit


        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )




    

    def forward(self, x: Tensor, y:Tensor) -> Union[tuple, Tensor]:
        """
        Estimate an RGB colour for the illuminant of the input image
        @param x: the image for which the colour of the illuminant has to be estimated
        @return: the colour estimate as a Tensor. If confidence-weighted pooling is used, the per-path colour estimates
        and the confidence weights are returned as well (used for visualizations)
        """
        x = self.backbone1(x)

        out = self.fc(x)
        
        pred1 = normalize(out, dim=1)
        indices = torch.arange(0, pred1.size(0), 3, dtype=torch.long)

        pred = pred1[indices+1]
        return pred