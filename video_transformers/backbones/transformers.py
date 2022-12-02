from typing import Dict

from video_transformers.backbones.base import Backbone
from video_transformers.utils.torch import unfreeze_last_n_stages as unfreeze_last_n_stages_torch

models_2d = ["convnext", "levit", "cvt", "clip", "swin", "vit", "deit", "beit", "resnet"]
models_3d = ["videomae", "timesformer"]


class TransformersBackbone(Backbone):
    def __init__(self, model_name: str, num_unfrozen_stages=0, **backbone_kwargs):
        super(TransformersBackbone, self).__init__()
        from transformers import AutoFeatureExtractor, AutoModel

        backbone = AutoModel.from_pretrained(model_name, **backbone_kwargs)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, **backbone_kwargs)

        if backbone.base_model_prefix == "clip":
            from transformers import CLIPVisionModel

            backbone = CLIPVisionModel.from_pretrained(model_name)

        mean, std = feature_extractor.image_mean, feature_extractor.image_std

        if hasattr(backbone.config, "hidden_size"):  # vit, swin, deit
            num_features = backbone.config.hidden_size
        elif hasattr(backbone.config, "hidden_sizes"):  # levit, convnext, resnet
            num_features = backbone.config.hidden_sizes[-1]
        elif hasattr(backbone.config, "embed_dim"):  # cvt
            num_features = backbone.config.embed_dim[-1]
        elif hasattr(backbone.config, "projection_dim"):  # clip
            num_features = backbone.config.projection_dim
        else:
            raise NotImplementedError(f"Huggingface model not supported: {backbone.base_model_prefix}")

        # set model input num frames
        if hasattr(backbone.config, "num_frames"):  # videomae
            self._num_frames = backbone.config.num_frames
        else:
            self._num_frames = 1

        self.model = backbone
        self.num_features = num_features
        self.mean = mean
        self.std = std
        self._model_name = model_name

        if self.model.base_model_prefix in models_2d:
            self._type = "2d_backbone"
        elif self.model.base_model_prefix in models_3d:
            self._type = "3d_backbone"
        else:
            raise NotImplementedError(f"Huggingface model not supported: {self.model.base_model_prefix}")

        self.unfreeze_last_n_stages(num_unfrozen_stages)

    @property
    def type(self) -> str:
        return self._type

    @property
    def framework(self) -> Dict:
        import transformers

        return {"name": "transformers", "version": transformers.__version__}

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def num_frames(self) -> int:
        return self._num_frames

    def forward(self, x):
        if self.model.base_model_prefix in models_3d:
            # ensure num_timesteps matches model num_frames
            if x.shape[2] != self.num_frames:
                raise ValueError(
                    f"Input has {x.shape[2]} frames, but {self.model_name} accepts {self.num_frames} frames. Set num_timesteps to {self.num_frames}."
                )
            # x: (B, C, T, H, W)
            x = x.permute(0, 2, 1, 3, 4)
            # x: (B, T, C, H, W)
            output = self.model(pixel_values=x, return_dict=True)[0]
            # output: batch x latent_dim x num_features
        else:
            output = self.model(pixel_values=x, return_dict=True)[1]
            # output: batch x 1 x num_features
        if output.dim() == 3:
            output = output.mean(1)
        return output  # output: batch x num_features

    def unfreeze_last_n_stages(self, n):
        if self.model.base_model_prefix == "convnext":
            stages = []
            # stages.append(self.model.base_model.embeddings)
            # freeze embeddings
            for param in self.model.base_model.embeddings.parameters():
                param.requires_grad = False
            # unfreeze last n stages
            stages.extend(self.model.base_model.encoder.stages)
            stages.append(self.model.base_model.layernorm)
            unfreeze_last_n_stages_torch(stages, n)
        elif self.model.base_model_prefix == "levit":
            stages = []
            # stages.extend(list(self.model.base_model.patch_embeddings.children()))
            # freeze embeddings
            for param in self.model.base_model.patch_embeddings.parameters():
                param.requires_grad = False
            # unfreeze last n stages
            stages.extend(self.model.base_model.encoder.stages)
            unfreeze_last_n_stages_torch(stages, n)
        elif self.model.base_model_prefix == "cvt":
            stages = []
            stages.extend(list(self.model.base_model.encoder.stages.children()))
            unfreeze_last_n_stages_torch(stages, n)
        elif self.model.base_model_prefix == "clip":
            stages = []
            # stages.extend(list(self.model.base_model.vision_model.embeddings.children()))
            # freeze embeddings
            for param in self.model.base_model.vision_model.embeddings.parameters():
                param.requires_grad = False
            # unfreeze last n stages
            stages.extend(list(self.model.base_model.vision_model.encoder.layers.children()))
            unfreeze_last_n_stages_torch(stages, n)
        elif self.model.base_model_prefix in ["swin", "vit", "deit", "beit"]:
            stages = []
            # stages.extend(list(self.model.base_model.embeddings.children()))
            # freeze embeddings
            for param in self.model.base_model.embeddings.parameters():
                param.requires_grad = False
            # unfreeze last n stages
            stages.extend(list(self.model.base_model.encoder.layers.children()))
            stages.append(self.model.base_model.layernorm)
            unfreeze_last_n_stages_torch(stages, n)
        elif self.model.base_model_prefix == "videomae":
            stages = []
            # freeze embeddings
            for param in self.model.base_model.embeddings.parameters():
                param.requires_grad = False
            # unfreeze last n stages
            stages.extend(list(self.model.base_model.encoder.layer.children()))
            unfreeze_last_n_stages_torch(stages, n)
        elif self.model.base_model_prefix == "timesformer":
            stages = []
            # freeze embeddings
            for param in self.model.base_model.embeddings.parameters():
                param.requires_grad = False
            # unfreeze last n stages
            stages.extend(list(self.model.base_model.encoder.layer.children()))
            unfreeze_last_n_stages_torch(stages, n)
        elif self.model.base_model_prefix == "resnet":
            stages = []
            # stages.append(self.model.base_model.embeddings)
            # freeze embeddings
            for param in self.model.base_model.embedder.parameters():
                param.requires_grad = False
            # unfreeze last n stages
            stages.extend(self.model.base_model.encoder.stages)
            unfreeze_last_n_stages_torch(stages, n)
        else:
            raise NotImplementedError(f"Freezing not supported for Huggingface model: {self.model.base_model_prefix}")
