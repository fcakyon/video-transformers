from typing import Dict, Union

from video_transformers.backbones.base import Backbone
from video_transformers.modules import TimeDistributed


class AutoBackbone:
    """
    AutoBackbone is a class that automatically instantiates a video model backbone from a config.
    """

    @classmethod
    def from_config(cls, config: Dict) -> Union[Backbone, TimeDistributed]:
        backbone_framework = config.get("framework")
        backbone_type = config.get("type")
        backbone_model_name = config.get("model_name")

        if backbone_framework["name"] == "transformers":
            from video_transformers.backbones.transformers import TransformersBackbone

            backbone = TransformersBackbone(model_name=backbone_model_name)
        elif backbone_framework["name"] == "timm":
            from video_transformers.backbones.timm import TimmBackbone

            backbone = TimmBackbone(model_name=backbone_model_name)
        else:
            raise ValueError(f"Unknown framework {backbone_framework}")

        if backbone_type == "2d_backbone":
            from video_transformers.modules import TimeDistributed

            backbone = TimeDistributed(backbone)
        return backbone
