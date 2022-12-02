from typing import Dict, Union

from video_transformers.backbones.base import Backbone
from video_transformers.modeling import TimeDistributed


class AutoBackbone:
    """
    AutoBackbone is a class that automatically instantiates a video model backbone from a config.
    """

    @classmethod
    def from_config(cls, config: Dict) -> Union[Backbone, TimeDistributed]:
        backbone_framework = config.get("framework")
        backbone_type = config.get("type")
        backbone_model_name = config.get("model_name")

        from video_transformers.backbones.transformers import TransformersBackbone

        backbone = TransformersBackbone(model_name=backbone_model_name)

        if backbone_type == "2d_backbone":
            from video_transformers.modeling import TimeDistributed

            backbone = TimeDistributed(backbone)
        return backbone

    @classmethod
    def from_transformers(cls, name_or_path: str) -> Union[Backbone, TimeDistributed]:
        from video_transformers.backbones.transformers import TransformersBackbone

        backbone = TransformersBackbone(model_name=name_or_path)

        if backbone.type == "2d_backbone":
            raise ValueError("2D backbones are not supported for from_transformers method.")
        return backbone
