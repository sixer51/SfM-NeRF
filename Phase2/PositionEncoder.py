import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def positionEncoder(input, L, includeInput=True):
    encodingRes = []
    if includeInput:
        encodingRes.extend(input)
    pi = torch.acos(torch.Tensor([-1]))

    for i in range(L):
        for x in input:
            sinRes = torch.sin(pi * x * 2.**i)
            cosRes = torch.cos(pi * x * 2.**i)
            encodingRes.extend([sinRes, cosRes])
    
    outputDim = len(encodingRes)
    return torch.Tensor(encodingRes), outputDim