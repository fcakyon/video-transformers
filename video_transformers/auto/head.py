from typing import Dict


class AutoHead:
    """
    AutoHead is a class that automatically instantiates a video model head from a config.
    """

    @classmethod
    def from_config(cls, config: Dict):
        head_class_name = config.get("name")
        if head_class_name == "LinearHead":
            from video_transformers.heads import LinearHead

            hidden_size = config.get("hidden_size")
            num_classes = config.get("num_classes")
            dropout_p = config.get("dropout_p")
            return LinearHead(hidden_size, num_classes, dropout_p)
        else:
            raise ValueError(f"Unsupported head class name: {head_class_name}")
