import torch
import torch.nn as nn
import numpy as np

class Nerf(nn.module):
    def __init__(self, depth=8, width=256, dimPos=3, dimDirc=3):
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
            nn.ReLU()
        )
        self.linearFeatureDensity = nn.Sequential(
            nn.Linear(self.width, self.width + 1),
            nn.ReLU()
        )
        self.linearLast = nn.Sequential(
            nn.Linear(self.width // 2, 3),
            nn.ReLU()
        )

    def forward(self, input):
        pos, dirc = torch.split(input, [self.dimPos, self.dimDirc], dim=-1)
        x = self.linear1(pos)
        for _ in range(self.depth-2):
            x = self.block(x)

        x = self.linearFeatureDensity(x)
        density = x[0]
        feature = x[1:]
        x = torch.cat([feature, dirc], -1)
        rgb = self.linearLast(x)
        output = torch.cat([rgb, density], -1)
        return output
