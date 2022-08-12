from typing import Dict, Tuple

import pytorchvideo.data
import torch
import torch.utils.data
from accelerate.logging import get_logger
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop, Compose, Lambda, RandomCrop, RandomHorizontalFlip

from video_transformers.utils.dataset import LabeledVideoDataset, LabeledVideoPaths
from video_transformers.utils.extra import class_to_config

logger = get_logger(__name__)


class VideoPreprocess:
    @classmethod
    def from_config(cls, config: Dict, **kwargs) -> "VideoPreprocess":
        """
        Creates an instance of the class from a config.

        Args:
            config: config dictionary
            kwargs: owerwrite config values
        """
        for ke, val in kwargs.items():
            config.update({ke, val})
        return cls(**config)

    def __init__(
        self,
        timesteps: int = 8,
        input_size: int = 224,
        means: Tuple[float] = (0.45, 0.45, 0.45),
        stds: Tuple[float] = (0.225, 0.225, 0.225),
        min_short_side_scale: int = 256,
        max_short_side_scale: int = 320,
        horizontal_flip_p: float = 0.5,
    ):
        """
        Creates preprocess transforms.

        Args:
            timesteps: number of frames in a video clip
            input_size: model input isze
            means: mean of the video clip
            stds: standard deviation of the video clip
            min_short_side_scale: minimum short side of the video clip
            max_short_side_scale: maximum short side of the video clip
            horizontal_flip_p: probability of horizontal flip

        Properties:
            train_transform: transforms for training
            train_video_transform: training transforms for video clips
            val_transform: transforms for validation
            val_video_transform: validation transforms for video clips

        """
        super().__init__()

        self.timesteps = timesteps
        self.input_size = input_size
        self.means = means
        self.stds = stds
        self.min_short_side_scale = min_short_side_scale
        self.max_short_side_scale = max_short_side_scale
        self.horizontal_flip_p = horizontal_flip_p

        # Transforms applied to train dataset.
        self.train_video_transform = Compose(
            [
                UniformTemporalSubsample(self.timesteps),
                Lambda(lambda x: x / 255.0),
                Normalize(self.means, self.stds),
                RandomShortSideScale(
                    min_size=self.min_short_side_scale,
                    max_size=self.max_short_side_scale,
                ),
                RandomCrop(self.input_size),
                RandomHorizontalFlip(p=self.horizontal_flip_p),
            ]
        )

        self.train_transform = ApplyTransformToKey(key="video", transform=self.train_video_transform)

        # Transforms applied on val dataset or for inference.
        self.val_video_transform = Compose(
            [
                UniformTemporalSubsample(self.timesteps),
                Lambda(lambda x: x / 255.0),
                Normalize(self.means, self.stds),
                ShortSideScale(self.min_short_side_scale),
                CenterCrop(self.input_size),
            ]
        )
        self.val_transform = ApplyTransformToKey(key="video", transform=self.val_video_transform)


class VideoDataModule:
    def __init__(
        self,
        train_root: str,
        val_root: str,
        test_root: str = None,
        clip_duration: int = 2,
        train_dataset_multiplier: int = 1,
        batch_size: int = 4,
        num_workers: int = 4,
        num_timesteps: int = 8,
        preprocess_input_size: int = 224,
        preprocess_means: Tuple[float] = (0.45, 0.45, 0.45),
        preprocess_stds: Tuple[float] = (0.225, 0.225, 0.225),
        preprocess_min_short_side_scale: int = 256,
        preprocess_max_short_side_scale: int = 320,
        preprocess_horizontal_flip_p: float = 0.5,
    ):
        """
        Initialize video data module.

        Folder structure:
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

        Args:
            train_root: str
                Path to kinetics formatted train folder.
            val_root: str
                Path to kinetics formatted train folder.
            test_root: str
                Path to kinetics formatted train folder.
            clip_duration: float
                Duration of sampled clip for each video.
            train_dataset_multiplier: int
                Multipler for number of of random training data samples.
            batch_size: int
                Batch size for training and validation.
            num_workers: int
                Number of parallel processes fetching data.
            num_timesteps: int
                Number of frames to subsample from each clip.
            preprocess_crop_size: int
                Size of the random crop.
            preprocess_means: Tuple[float]
                Mean pixel value to be used during normalization.
            preprocess_stds: Tuple[float]
                Standard deviation pixel value to be used during normalization.
            preprocess_min_short_side_scale: int
                Minimum value of the short side of the clip after resizing.
            preprocess_max_short_side_scale: int
                Maximum value of the short side of the clip after resizing.
            preprocess_horizontal_flip_p: float
                Probability of horizontal flip.
        """
        self.preprocess_config = {
            "timesteps": num_timesteps,
            "input_size": preprocess_input_size,
            "means": preprocess_means,
            "stds": preprocess_stds,
            "min_short_side_scale": preprocess_min_short_side_scale,
            "max_short_side_scale": preprocess_max_short_side_scale,
            "horizontal_flip_p": preprocess_horizontal_flip_p,
        }
        self.preprocess = VideoPreprocess.from_config(self.preprocess_config)

        self.dataloader_config = {"batch_size": batch_size, "num_workers": num_workers, "clip_duration": clip_duration}

        self.train_root = train_root
        self.val_root = val_root
        self.test_root = test_root if test_root is not None else val_root
        self.train_dataset_multiplier = train_dataset_multiplier
        self.labels = None

        self.train_dataloader = self._get_train_dataloader()
        self.val_dataloader = self._get_val_dataloader()
        self.test_dataloader = self._get_test_dataloader()

    @property
    def num_classes(self):
        return len(self.labels)

    @property
    def config(self) -> Dict:
        return class_to_config(self, ignored_attrs=("config", "train_root", "val_root", "test_root"))

    def _get_train_dataloader(self):
        labeled_video_paths = LabeledVideoPaths.from_path(self.train_root)
        labeled_video_paths.path_prefix = ""
        video_sampler = torch.utils.data.RandomSampler
        clip_sampler = pytorchvideo.data.make_clip_sampler("random", self.dataloader_config["clip_duration"])
        dataset = LabeledVideoDataset(
            labeled_video_paths,
            clip_sampler,
            video_sampler,
            self.preprocess.train_transform,
            decode_audio=False,
            decoder="pyav",
            dataset_multiplier=self.train_dataset_multiplier,
        )
        self.labels = dataset.labels
        return DataLoader(
            dataset,
            batch_size=self.dataloader_config["batch_size"],
            num_workers=self.dataloader_config["num_workers"],
            drop_last=False,
        )

    def _get_val_dataloader(self):
        labeled_video_paths = LabeledVideoPaths.from_path(self.val_root)
        labeled_video_paths.path_prefix = ""
        video_sampler = torch.utils.data.SequentialSampler
        clip_sampler = pytorchvideo.data.make_clip_sampler("uniform", self.dataloader_config["clip_duration"])
        dataset = LabeledVideoDataset(
            labeled_video_paths,
            clip_sampler,
            video_sampler,
            self.preprocess.val_transform,
            decode_audio=False,
            decoder="pyav",
        )
        return DataLoader(
            dataset,
            batch_size=self.dataloader_config["batch_size"],
            num_workers=self.dataloader_config["num_workers"],
            drop_last=False,
        )

    def _get_test_dataloader(self):
        labeled_video_paths = LabeledVideoPaths.from_path(self.test_root)
        labeled_video_paths.path_prefix = ""
        video_sampler = torch.utils.data.SequentialSampler
        clip_sampler = pytorchvideo.data.make_clip_sampler("uniform", self.dataloader_config["clip_duration"])
        dataset = LabeledVideoDataset(
            labeled_video_paths,
            clip_sampler,
            video_sampler,
            self.preprocess.val_transform,
            decode_audio=False,
            decoder="pyav",
        )
        return DataLoader(
            dataset,
            batch_size=self.dataloader_config["batch_size"],
            num_workers=self.dataloader_config["num_workers"],
            drop_last=False,
        )
