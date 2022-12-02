import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from huggingface_hub.constants import PYTORCH_WEIGHTS_NAME
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.hub_mixin import PyTorchModelHubMixin
from torch import nn

import video_transformers.backbones.base
import video_transformers.predict
from video_transformers.heads import LinearHead
from video_transformers.hfhub_wrapper.hub_mixin import push_to_hub
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


class VideoModel(nn.Module, PyTorchModelHubMixin):
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
    def from_config(cls, config: Dict) -> "VideoModel":
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
        if config["neck"] is not None:
            neck = video_transformers.AutoNeck.from_config(config["neck"])

        return cls(
            backbone=backbone,
            head=head,
            neck=neck,
            preprocessor_config=config["preprocessor"],
            labels=config["labels"],
            task=config["task"],
        )

    @classmethod
    def from_transformers(cls, name_of_path: str, clip_duration: int = 2) -> "VideoModel":
        """
        Loads a model from a hf/transformers models.
        """
        from transformers import AutoConfig, AutoProcessor

        import video_transformers
        import video_transformers.data

        processor = AutoProcessor.from_pretrained(name_of_path)
        model_config = AutoConfig.from_pretrained(name_of_path)
        labels = list(model_config.id2label.values())

        video_preprocessor_config = {
            "num_timesteps": model_config.num_frames,
            "input_size": model_config.image_size,
            "means": [0.45, 0.45, 0.45],
            "stds": [0.225, 0.225, 0.225],
            "min_short_side": model_config.image_size,
            "max_short_side": model_config.image_size,
            "horizontal_flip_p": 0,
            "clip_duration": clip_duration,
        }

        backbone = video_transformers.AutoBackbone.from_transformers(name_of_path)
        head = video_transformers.AutoHead.from_transformers(name_of_path)

        return cls(
            backbone=backbone,
            head=head,
            neck=None,
            preprocessor_config=video_preprocessor_config,
            labels=labels,
            task="single_label_classification",
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

    def push_to_hub(
        self,
        *,
        repo_path_or_name: Optional[str] = None,
        repo_url: Optional[str] = None,
        commit_message: Optional[str] = "Add model",
        organization: Optional[str] = None,
        private: bool = False,
        api_endpoint: Optional[str] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        git_user: Optional[str] = None,
        git_email: Optional[str] = None,
        config: Optional[dict] = None,
        skip_lfs_files: bool = False,
        repo_id: Optional[str] = None,
        token: Optional[str] = None,
        branch: Optional[str] = None,
        create_pr: Optional[bool] = None,
        allow_patterns: Optional[Union[List[str], str]] = None,
        ignore_patterns: Optional[Union[List[str], str]] = None,
    ) -> str:
        """
        Upload model checkpoint to the Hub.

        Use `allow_patterns` and `ignore_patterns` to precisely filter which files
        should be pushed to the hub. See [`upload_folder`] reference for more details.

        Parameters:
            repo_id (`str`, *optional*):
                Repository name to which push.
            commit_message (`str`, *optional*):
                Message to commit while pushing.
            private (`bool`, *optional*, defaults to `False`):
                Whether the repository created should be private.
            api_endpoint (`str`, *optional*):
                The API endpoint to use when pushing the model to the hub.
            token (`str`, *optional*):
                The token to use as HTTP bearer authorization for remote files.
                If not set, will use the token set when logging in with
                `transformers-cli login` (stored in `~/.huggingface`).
            branch (`str`, *optional*):
                The git branch on which to push the model. This defaults to
                the default branch as specified in your repository, which
                defaults to `"main"`.
            create_pr (`boolean`, *optional*):
                Whether or not to create a Pull Request from `branch` with that commit.
                Defaults to `False`.
            config (`dict`, *optional*):
                Configuration object to be saved alongside the model weights.
            allow_patterns (`List[str]` or `str`, *optional*):
                If provided, only files matching at least one pattern are pushed.
            ignore_patterns (`List[str]` or `str`, *optional*):
                If provided, files matching any of the patterns are not pushed.

        Returns:
            The url of the commit of your model in the given repository.
        """
        return push_to_hub(
            self=self,
            repo_path_or_name=repo_path_or_name,
            repo_url=repo_url,
            commit_message=commit_message,
            organization=organization,
            private=private,
            api_endpoint=api_endpoint,
            use_auth_token=use_auth_token,
            git_user=git_user,
            git_email=git_email,
            config=config,
            skip_lfs_files=skip_lfs_files,
            repo_id=repo_id,
            token=token,
            branch=branch,
            create_pr=create_pr,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    def __init__(
        self,
        backbone: Union[TimeDistributed, video_transformers.backbones.base.Backbone],
        head: LinearHead,
        neck=None,
        labels: List[str] = None,
        preprocessor_config: Dict = None,
        task: str = None,
    ):
        """
        Args:
            backbone: Backbone model.
            head: Head model.
            neck: Neck model.
            labels: List of labels (required for onnx export and predict).
            preprocessor_config: Preprocessor config (required for onnx export and predict).
            task: Task name (required for predict).
        """
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

        # required for exporting to ONNX and predict
        self.preprocessor_config = preprocessor_config
        self.labels = labels

        self.task = task

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
        config["task"] = self.task
        config["backbone"] = self.backbone.config
        config["head"] = self.head.config
        if self.neck is not None:
            config["neck"] = self.neck.config
        else:
            config["neck"] = None
        config["labels"] = self.labels
        config["preprocessor"] = self.preprocessor_config
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
        import video_transformers.deployment.onnx

        video_transformers.deployment.onnx.export(self, quantize, opset_version, export_dir, export_filename)

    def to_gradio(
        self,
        examples: List[str],
        author_username: str = None,
        export_dir: str = "runs/exports/",
        export_filename: str = "app.py",
    ):
        """
        Export model as Gradio App.

        Args:
            examples: List of examples to use for the app.
            author_username: Author's username.
            export_dir: Directory to export model to.
            export_filename: Filename to export model to.
        """
        import video_transformers.deployment.gradio

        video_transformers.deployment.gradio.export_gradio_app(
            self, examples, author_username, export_dir, export_filename
        )

    @property
    def example_input_array(self):
        if self.preprocessor_config:
            return torch.rand(
                1,
                3,
                self.preprocessor_config["num_timesteps"],
                self.preprocessor_config["input_size"],
                self.preprocessor_config["input_size"],
            )
        else:
            return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: BxCxTxHxW
        x = self.backbone(x)
        if self.neck is not None:
            x = self.neck(x)
        x = self.head(x)
        return x

    def predict(self, video_or_folder_path: Union[str, Path], mode: str = "first_clip") -> List[Dict]:
        """
        Predict the labels and probabilities of a video or folder of videos.
        Supports local file path/folder directory, S3 URI and https URL.

        Args:
            model: The model to use for prediction.
            video_or_folder_path: The path to the video or folder of videos.
                Supports local file path/folder directory, S3 URI and https URL.
            mode: The mode to use for prediction. Can be "first_clip", "average_all", "random_clip", "uniform_clip".
        """
        result = video_transformers.predict.predict(
            self,
            video_or_folder_path=video_or_folder_path,
            mode=mode,
            preprocessor_config=self.preprocessor_config,
            labels=self.labels,
            device=next(self.parameters()).device,
        )

        return result


if __name__ == "__main__":
    VideoModel.from_transformers("facebook/timesformer-base-finetuned-k600")
