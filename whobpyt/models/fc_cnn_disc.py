""" 
Light-weight 2D CNN discriminator on static FC matrices
Call with shape [b, 1, N, N] -> prob. real \in [0, 1]
"""

import torch, torch.nn as nn
from typing import Tuple

class FCCNNDisc(nn.Module):
    def __init__(self, n_nodes: int, p_drop: float = .3):
        super().__init__()
        # encoder
        self.feat = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 5, padding=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 5, padding=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AvgPool2d(2)                                      # (N/2 × N/2)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)               # (B,32,1,1)

        # classifier head
        self.clf = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 256), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(256, 1),  nn.Sigmoid()
        )

        self.apply(self._init)

    def forward(self, fc: torch.Tensor) -> torch.Tensor:
        """
        fc : (B, N, N) or (B, 1, N, N) float32/64
        |
        prob_real : (B, 1) in [0,1]
        """
        if fc.ndim == 3:                     # (B,1,N,N)
            fc = fc.unsqueeze(1)
        z = self.feat(fc)                    # (B,32,N/2,N/2)
        z = self.global_pool(z)              # (B,32,1,1)
        prob = self.clf(z)
        return prob                          # (B,1)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0.)


class LightFCCNN(nn.Module):
    """ lower-capacity head + hinge output (no sigmoid) """
    def __init__(self, n_nodes):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(1,16,5,padding=2), nn.InstanceNorm2d(16), nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
            nn.Conv2d(16,32,5,padding=2), nn.InstanceNorm2d(32), nn.LeakyReLU(0.2),
        )
        self.clf  = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32,32), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(32,1)              # raw score
        )
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight)

    def forward(self, x):               # x : (B,1,N,N)
        return self.clf(self.feat(x))  # (B,1) (no sigmoid)
