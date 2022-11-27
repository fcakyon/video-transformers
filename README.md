<p align="center">
<img src="https://user-images.githubusercontent.com/34196005/180642397-1f56d9c7-dee2-48d4-acbf-c3bc62f36150.png" width="500">
</p>

<p align="center">
    Easiest way of fine-tuning HuggingFace video classification models.
</p>

<div align="center">
    <a href="https://badge.fury.io/py/video-transformers"><img src="https://badge.fury.io/py/video-transformers.svg" alt="pypi version"></a>
    <a href="https://pepy.tech/project/video-transformers"><img src="https://pepy.tech/badge/video-transformers" alt="total downloads"></a>
    <a href="https://twitter.com/fcakyon"><img src="https://img.shields.io/twitter/follow/fcakyon?color=blue&logo=twitter&style=flat" alt="fcakyon twitter"></a>
</div>

## 🚀 Features

`video-transformers` uses:

- 🤗 [accelerate](https://github.com/huggingface/accelerate) for distributed training,

- 🤗 [evaluate](https://github.com/huggingface/evaluate) for evaluation,

- [pytorchvideo](https://github.com/facebookresearch/pytorchvideo) for dataloading

and supports:

- creating and fine-tunining video models using [transformers](https://github.com/huggingface/transformers) and [timm](https://github.com/rwightman/pytorch-image-models) vision models

- experiment tracking with [neptune](https://neptune.ai/), [tensorboard](https://www.tensorflow.org/tensorboard) and other trackers

- exporting fine-tuned models in [ONNX](https://onnx.ai/) format

- pushing fine-tuned models into [HuggingFace Hub](https://huggingface.co/models?pipeline_tag=image-classification&sort=downloads)

- loading pretrained models from [HuggingFace Hub](https://huggingface.co/models?pipeline_tag=image-classification&sort=downloads)

- Automated [Gradio app](https://gradio.app/), and [space](https://huggingface.co/spaces) creation 

## 🏁 Installation

- Install `Pytorch`:

```bash
conda install pytorch=1.11.0 torchvision=0.12.0 cudatoolkit=11.3 -c pytorch
```

- Install pytorchvideo from main branch:

```bash
pip install git+https://github.com/facebookresearch/pytorchvideo.git
```

- Install `video-transformers`:

```bash
pip install video-transformers
```

## 🔥 Usage

- Prepare video classification dataset in such folder structure (.avi and .mp4 extensions are supported):

```bash
train_root
    label_1
        video_1
        video_2
        ...
    label_2
        video_1
        video_2
        ...
    ...
val_root
    label_1
        video_1
        video_2
        ...
    label_2
        video_1
        video_2
        ...
    ...
```

- Fine-tune CVT (from HuggingFace) + Transformer based video classifier:

```python
from torch.optim import AdamW
from video_transformers import TimeDistributed, VideoModel
from video_transformers.backbones.transformers import TransformersBackbone
from video_transformers.data import VideoDataModule
from video_transformers.heads import LinearHead
from video_transformers.necks import TransformerNeck
from video_transformers.trainer import trainer_factory
from video_transformers.utils.file import download_ucf6

backbone = TimeDistributed(TransformersBackbone("microsoft/cvt-13", num_unfrozen_stages=0))
neck = TransformerNeck(
    num_features=backbone.num_features,
    num_timesteps=8,
    transformer_enc_num_heads=4,
    transformer_enc_num_layers=2,
    dropout_p=0.1,
)

download_ucf6("./")
datamodule = VideoDataModule(
    train_root="ucf6/train",
    val_root="ucf6/val",
    batch_size=4,
    num_workers=4,
    num_timesteps=8,
    preprocess_input_size=224,
    preprocess_clip_duration=1,
    preprocess_means=backbone.mean,
    preprocess_stds=backbone.std,
    preprocess_min_short_side=256,
    preprocess_max_short_side=320,
    preprocess_horizontal_flip_p=0.5,
)

head = LinearHead(hidden_size=neck.num_features, num_classes=datamodule.num_classes)
model = VideoModel(backbone, head, neck)

optimizer = AdamW(model.parameters(), lr=1e-4)

Trainer = trainer_factory("single_label_classification")
trainer = Trainer(
    datamodule,
    model,
    optimizer=optimizer,
    max_epochs=8
)

trainer.fit()

```

- Fine-tune MobileViT (from Timm) + GRU based video classifier:

```python
from video_transformers import TimeDistributed, VideoModel
from video_transformers.backbones.timm import TimmBackbone
from video_transformers.data import VideoDataModule
from video_transformers.heads import LinearHead
from video_transformers.necks import GRUNeck
from video_transformers.trainer import trainer_factory
from video_transformers.utils.file import download_ucf6

backbone = TimeDistributed(TimmBackbone("mobilevitv2_100", num_unfrozen_stages=0))
neck = GRUNeck(num_features=backbone.num_features, hidden_size=128, num_layers=2, return_last=True)

download_ucf6("./")
datamodule = VideoDataModule(
    train_root="ucf6/train",
    val_root="ucf6/val",
    batch_size=4,
    num_workers=4,
    num_timesteps=8,
    preprocess_input_size=224,
    preprocess_clip_duration=1,
    preprocess_means=backbone.mean,
    preprocess_stds=backbone.std,
    preprocess_min_short_side=256,
    preprocess_max_short_side=320,
    preprocess_horizontal_flip_p=0.5,
)

head = LinearHead(hidden_size=neck.hidden_size, num_classes=datamodule.num_classes)
model = VideoModel(backbone, head, neck)

Trainer = trainer_factory("single_label_classification")
trainer = Trainer(
    datamodule,
    model,
    max_epochs=8
)

trainer.fit()

```

- Perform prediction for a single file or folder of videos:

```python
from video_transformers import VideoModel

model = VideoModel.from_pretrained(model_name_or_path)

model.predict(video_path="video.mp4")
>> [{'filename': "video.mp4", 'predictions': {'class1': 0.98, 'class2': 0.02}}]
```


## 🤗 Full HuggingFace Integration

- Push your fine-tuned model to the hub:

```python
from video_transformers import VideoModel

model = VideoModel.from_pretrained("runs/exp/checkpoint")

model.push_to_hub('model_name')
```

- Load any pretrained video-transformer model from the hub:

```python
from video_transformers import VideoModel

model = VideoModel.from_pretrained("runs/exp/checkpoint")

model.from_pretrained('account_name/model_name')
```

- Push your model to HuggingFace hub with auto-generated model-cards:

```python
from video_transformers import VideoModel

model = VideoModel.from_pretrained("runs/exp/checkpoint")
model.push_to_hub('account_name/app_name')
```

- (Incoming feature) Push your model as a Gradio app to HuggingFace Space:

```python
from video_transformers import VideoModel

model = VideoModel.from_pretrained("runs/exp/checkpoint")
model.push_to_space('account_name/app_name')
```

## 📈 Multiple tracker support

- Tensorboard tracker is enabled by default.

- To add Neptune/Layer ... tracking:

```python
from video_transformers.tracking import NeptuneTracker
from accelerate.tracking import WandBTracker

trackers = [
    NeptuneTracker(EXPERIMENT_NAME, api_token=NEPTUNE_API_TOKEN, project=NEPTUNE_PROJECT),
    WandBTracker(project_name=WANDB_PROJECT)
]

trainer = Trainer(
    datamodule,
    model,
    trackers=trackers
)

```

## 🕸️ ONNX support

- Convert your trained models into ONNX format for deployment:

```python
from video_transformers import VideoModel

model = VideoModel.from_pretrained("runs/exp/checkpoint")
model.to_onnx(quantize=False, opset_version=12, export_dir="runs/exports/", export_filename="model.onnx")
```

## 🤗 Gradio support

- Convert your trained models into Gradio App for deployment:

```python
from video_transformers import VideoModel

model = VideoModel.from_pretrained("runs/exp/checkpoint")
model.to_gradio(examples=['video.mp4'], export_dir="runs/exports/", export_filename="app.py")
```
