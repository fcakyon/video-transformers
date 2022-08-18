from typing import Tuple

from torch import nn

from video_transformers.backbones.base import Backbone
from video_transformers.modeling import Identity
from video_transformers.utils.torch import unfreeze_last_n_stages as unfreeze_last_n_stages_torch


class TimmBackbone(Backbone):
    def __init__(self, model_name: str, pretrained: bool = False, num_unfrozen_stages=0, **backbone_kwargs):
        super(TimmBackbone, self).__init__()
        import timm

        backbone = timm.create_model(model_name, pretrained=pretrained, **backbone_kwargs)
        if backbone.pretrained_cfg["classifier"] == "head.fc":
            backbone.head.fc = Identity()
        elif backbone.pretrained_cfg["classifier"] == "fc":  # adv_inception_v3
            backbone.fc = Identity()
        elif backbone.pretrained_cfg["classifier"] == "head":  # deit
            backbone.head = Identity()
        elif backbone.pretrained_cfg["classifier"] == "classifier":
            backbone.classifier = Identity()
        elif backbone.pretrained_cfg["classifier"] == ("head.l", "head_dist.l"):  # levit
            backbone = nn.Sequential(*list(backbone.children())[:-2])
        else:
            raise NotImplementedError(f"Backbone not supported: {backbone.pretrained_cfg}")

        mean, std = backbone.pretrained_cfg["mean"], backbone.pretrained_cfg["std"]
        num_features = backbone.num_features

        self.model = backbone
        self.num_features = num_features
        self.mean = mean
        self.std = std
        self._model_name = model_name
        self._type = "2d_backbone"

        self.unfreeze_last_n_stages(num_unfrozen_stages)

    @property
    def type(self):
        return self._type

    @property
    def framework(self):
        import timm

        return {"name": "timm", "version": timm.__version__}

    @property
    def model_name(self):
        return self._model_name

    def forward(self, x):
        return self.model(x)

    def unfreeze_last_n_stages(self, n):
        stages = [stage for stage in self.model.stem.children()] + [stage for stage in self.model.stages.children()]
        unfreeze_last_n_stages_torch(stages, n)
