from typing import Dict

import torch
from torch import nn

from video_transformers.utils.extra import class_to_config


class LinearHead(nn.Module):
    """
     (BxF)
       ↓
    Dropout
       ↓
    Linear
    """

    def __init__(self, hidden_size: int, num_classes: int, dropout_p: float = 0.0):
        super(LinearHead, self).__init__()

        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout_p = dropout_p

        self.dropout = nn.Dropout(dropout_p) if dropout_p != 0 else None
        self.linear = nn.Linear(in_features=hidden_size, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: BxF
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.linear(x)
        return x

    @property
    def config(self) -> Dict:
        return class_to_config(self)
