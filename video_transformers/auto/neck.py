from typing import Dict

from video_transformers.necks import BaseNeck


class AutoNeck:
    """
    AutoNeck is a class that automatically instantiates a video model neck from a config.

    """

    @classmethod
    def from_config(cls, config: Dict) -> BaseNeck:
        neck_class_name = config.get("name")
        if neck_class_name == "LSTMNeck":
            from video_transformers.necks import LSTMNeck

            num_features = config.get("num_features")
            hidden_size = config.get("hidden_size")
            num_layers = config.get("num_layers")
            return_last = config.get("return_last")
            return LSTMNeck(num_features, hidden_size, num_layers, return_last)
        elif neck_class_name == "GRUNeck":
            from video_transformers.necks import GRUNeck

            num_features = config.get("num_features")
            hidden_size = config.get("hidden_size")
            num_layers = config.get("num_layers")
            return_last = config.get("return_last")
            return GRUNeck(num_features, hidden_size, num_layers, return_last)
        elif neck_class_name == "TransformerNeck":
            from video_transformers.necks import TransformerNeck

            num_features = config.get("num_features")
            num_timesteps = config.get("num_timesteps")
            transformer_enc_num_heads = config.get("transformer_enc_num_heads")
            transformer_enc_num_layers = config.get("transformer_enc_num_layers")
            transformer_enc_act = config.get("transformer_enc_act")
            dropout_p = config.get("dropout_p")
            return_mean = config.get("return_mean")
            return TransformerNeck(
                num_features,
                num_timesteps,
                transformer_enc_num_heads,
                transformer_enc_num_layers,
                transformer_enc_act,
                dropout_p,
                return_mean,
            )

        else:
            raise ValueError(f"Unsupported neck class name: {neck_class_name}")
