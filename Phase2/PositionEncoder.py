import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def positionEncoder(input, L, includeInput=True):
    encodingRes = []
    if includeInput:
        encodingRes.append(input)
    # pi = torch.acos(torch.Tensor([-1]))

    for i in range(L):
        for fun in [torch.sin, torch.cos]:
            encodingRes.append(fun((2.0**i)*input))
    
    # outputDim = len(encodingRes)
    return torch.concat(encodingRes, axis=-1)