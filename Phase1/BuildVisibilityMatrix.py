'''
construct I x J binary visibility matrix V
j - point number
i - camera number
'''

import numpy as np

def BuildVisibilityMatrix(imgIDList, imagePointIdxList, visMap):
    visMatrix = visMap[imagePointIdxList, :]
    visMatrix = visMatrix[:, imgIDList].T
    return visMatrix.astype(int)


