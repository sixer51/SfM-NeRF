import torch
import torch.nn as nn
import numpy as np

class NeRFmodel(nn.Module):
    def __init__(self, depth=8, width=256, dimPos=3, dimDirc=3):
        super(NeRFmodel, self).__init__()
        self.depth = depth
        self.width = width
        self.dimPos = dimPos
        self.dimDirc = dimDirc
        self.linear1 = nn.Sequential(
            nn.Linear(self.dimPos, self.width),
            nn.ReLU()
        )
        self.block = nn.Sequential(
            nn.Linear(self.width, self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.width),
            nn.ReLU()
        )
        self.linearToFeature = nn.Sequential(
            nn.Linear(self.width, self.width)
        )
        self.linearToDensity = nn.Sequential(
            nn.Linear(self.width, 1)
        )
        self.linearFromFeatureDensity = nn.Sequential(
            nn.Linear(self.width + dimDirc, self.width // 2),
            nn.ReLU()
        )
        self.linearLast = nn.Sequential(
            nn.Linear(self.width // 2, 3)
        )

    def forward(self, pos, dirc):
        x = self.linear1(pos)
        x = self.block(x)

        feature = self.linearToFeature(x)
        density = self.linearToDensity(x)
        x = torch.cat([feature, dirc], -1)
        x = self.linearFromFeatureDensity(x)
        rgb = self.linearLast(x)
        density = density.view(-1, 1)
        output = torch.cat([rgb, density], -1)
        return output
