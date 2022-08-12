import json
import os
from typing import Dict, List, Union

import torch
from huggingface_hub.constants import PYTORCH_WEIGHTS_NAME
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.hub_mixin import PyTorchModelHubMixin
from torch import nn

import video_transformers.backbones.base
from video_transformers.deployment.onnx import export
from video_transformers.heads import LinearHead
from video_transformers.utils.torch import get_num_total_params, get_num_trainable_params


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class TimeDistributed(nn.Module):
    """
    Time x Backbone2D (BxCxTxHxW)
           ↓
         Pool2D
           ↓
        (BxTxF)
    """

    def __init__(self, backbone: video_transformers.backbones.base.Backbone, low_memory=False):
        super(TimeDistributed, self).__init__()
        self.backbone = backbone
        self.low_memory = low_memory
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    @property
    def num_features(self):
        return self.backbone.num_features

    @property
    def mean(self):
        return self.backbone.mean

    @property
    def std(self):
        return self.backbone.std

    @property
    def type(self):
        return self.backbone.type

    @property
    def model_name(self):
        return self.backbone.model_name

    @property
    def framework(self):
        return self.backbone.framework

    @property
    def num_trainable_params(self):
        return self.backbone.num_trainable_params

    @property
    def num_total_params(self):
        return self.backbone.num_total_params

    @property
    def config(self):
        return self.backbone.config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: BxCxTxHxW
        batch_size, num_channels, num_timesteps, height, width = x.size()

        if self.low_memory:
            output_matrix = []
            for timestep in range(num_timesteps):
                x_t = self.backbone(x[:, :, timestep, :, :])  # x_t: BxC
                output_matrix.append(x_t)
            x = torch.stack(output_matrix, dim=1)  # x: BxTxF
            # clear up memory space
            x_t = None
            output_matrix = None
        else:
            x = x.permute((0, 2, 1, 3, 4))  # x: BxTxCxHxW
            x = x.contiguous().view(batch_size * num_timesteps, num_channels, height, width)  # x: (BxT)xCxHxW
            x = self.backbone(x)  # x: (BxT)xF
            x = x.contiguous().view(batch_size, num_timesteps, x.size(1))  # x: BxTxF
        return x


class VideoClassificationModel(nn.Module, PyTorchModelHubMixin):
    """
    (BxCxTxHxW)
         ↓
      Backbone
         ↓
       Neck
         ↓
       (BxF)
         ↓
       Head

    """

    @classmethod
    def from_config(cls, config: Dict) -> "VideoClassificationModel":
        """
        Loads a model from a config file.
        """
        import video_transformers

        if isinstance(config, str):
            with open(config) as f:
                config = json.load(f)

        backbone = video_transformers.AutoBackbone.from_config(config["backbone"])
        head = video_transformers.AutoHead.from_config(config["head"])
        neck = None
        if "neck" in config:
            neck = video_transformers.AutoNeck.from_config(config["neck"])

        return cls(
            backbone=backbone,
            head=head,
            neck=neck,
            timesteps=config["num_timesteps"],
            input_size=config["preprocess_input_size"],
            labels=config["labels"],
        )

    @classmethod
    def _from_pretrained(
        cls,
        model_id,
        revision,
        cache_dir,
        force_download,
        proxies,
        resume_download,
        local_files_only,
        use_auth_token,
        map_location="cpu",
        strict=False,
        **model_kwargs,
    ):

        map_location = torch.device(map_location)

        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, PYTORCH_WEIGHTS_NAME)
        else:
            model_file = hf_hub_download(
                repo_id=model_id,
                filename=PYTORCH_WEIGHTS_NAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                use_auth_token=use_auth_token,
                local_files_only=local_files_only,
            )
        model = cls.from_config(**model_kwargs)

        state_dict = torch.load(model_file, map_location=map_location)
        model.load_state_dict(state_dict, strict=strict)
        model.eval()

        return model

    def __init__(
        self,
        backbone: Union[TimeDistributed, video_transformers.backbones.base.Backbone],
        head: LinearHead,
        neck=None,
        timesteps: int = None,
        input_size: int = None,
        labels: List[str] = None,
    ):
        """
        Args:
            backbone: Backbone model.
            head: Head model.
            neck: Neck model.
            timesteps: Number of input timesteps (required for onnx export).
            input_size: Input size of model (required for onnx export).
            labels: List of labels (required for onnx export).
        """
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

        # required for exporting to ONNX
        self.timesteps = timesteps
        self.input_size = input_size
        self.labels = labels

    @property
    def num_features(self):
        return self.backbone.num_features

    @property
    def mean(self):
        return self.backbone.mean

    @property
    def std(self):
        return self.backbone.std

    @property
    def backbone_type(self):
        return self.backbone.type

    @property
    def backbone_name(self):
        return self.backbone.model_name

    @property
    def backbone_framework(self):
        return self.backbone.framework

    @property
    def num_trainable_params(self):
        return get_num_trainable_params(self)

    @property
    def num_total_params(self):
        return get_num_total_params(self)

    @property
    def config(self):
        config = {}
        config["backbone"] = self.backbone.config
        config["head"] = self.head.config
        if self.neck is not None:
            config["neck"] = self.neck.config
        return config

    def to_onnx(
        self,
        quantize: bool = False,
        opset_version: int = 12,
        export_dir: str = "exports/",
        export_filename: str = "model.onnx",
    ):
        """
        Export model to ONNX format.

        Args:
            quantize: Whether to quantize the model.
            opset_version: Opset version to use.
            export_dir: Directory to export model to.
            export_filename: Filename to export model to.
        """

        export(self, quantize, opset_version, export_dir, export_filename)

    @property
    def example_input_array(self):
        if self.timesteps and self.input_size:
            return torch.rand(1, 3, self.timesteps, self.input_size, self.input_size)
        else:
            return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: BxCxTxHxW
        x = self.backbone(x)
        if self.neck is not None:
            x = self.neck(x)
        x = self.head(x)
        return x
