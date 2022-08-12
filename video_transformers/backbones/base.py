import inspect
from typing import Dict

from torch import nn

from video_transformers.utils.extra import class_to_config
from video_transformers.utils.torch import get_num_total_params, get_num_trainable_params


class Backbone(nn.Module):
    def __init__(
        self,
    ):
        super(Backbone, self).__init__()

    def forward(self, x):
        raise NotImplementedError()

    def unfreeze_last_n_stages(self, n):
        raise NotImplementedError()

    @property
    def num_trainable_params(self):
        return get_num_trainable_params(self.model)

    @property
    def num_total_params(self):
        return get_num_total_params(self.model)

    @property
    def type(self) -> str:
        NotImplementedError()  # 2d_backbone, 3d_backbone

    @property
    def framework(self) -> Dict:
        NotImplementedError()

    @property
    def model_name(self) -> str:
        NotImplementedError()

    @property
    def config(self) -> Dict:
        return class_to_config(self)
