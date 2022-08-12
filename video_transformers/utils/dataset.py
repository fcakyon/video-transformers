# modified from https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/data/labeled_video_paths.py
# and https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/data/labeled_video_dataset.py

from __future__ import annotations

import logging
import os
import pathlib
import zipfile
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
from iopath.common.file_io import g_pathmgr
from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data.labeled_video_dataset import LabeledVideoDataset as LabeledVideoDataset_
from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths as LabeledVideoPaths_
from pytorchvideo.data.labeled_video_paths import make_dataset_from_video_folders
from pytorchvideo.data.video import VideoPathHandler
from torchvision.datasets.folder import make_dataset

from video_transformers.utils.file import download_file

logger = logging.getLogger(__name__)


class LabeledVideoPaths(LabeledVideoPaths_):
    """
    LabeledVideoPaths contains pairs of video path and integer index label.
    """

    labels = None

    @classmethod
    def from_path(cls, data_path: str) -> LabeledVideoPaths:
        """
        Factory function that creates a LabeledVideoPaths object depending on the path
        type.
        - If it is a directory path it uses the LabeledVideoPaths.from_directory function.
        - If it's a file it uses the LabeledVideoPaths.from_csv file.
        Args:
            file_path (str): The path to the file to be read.
        """

        if g_pathmgr.isfile(data_path):
            return LabeledVideoPaths.from_csv(data_path)
        elif g_pathmgr.isdir(data_path):
            class_0 = g_pathmgr.ls(data_path)[0]
            video_0 = g_pathmgr.ls(pathlib.Path(data_path) / class_0)[0]
            video_0_path = pathlib.Path(data_path) / class_0 / video_0
            if g_pathmgr.isfile(video_0_path):
                return LabeledVideoPaths.from_directory(data_path)
            else:
                return LabeledVideoPaths.from_directory_of_video_folders(data_path)
        else:
            raise FileNotFoundError(f"{data_path} not found.")

    @classmethod
    def from_directory(cls, dir_path: str) -> LabeledVideoPaths:
        """
        Factory function that creates a LabeledVideoPaths object by parsing the structure
        of the given directory's subdirectories into the classification labels. It
        expects the directory format to be the following:
             dir_path/<class_name>/<video_name>.mp4

        Classes are indexed from 0 to the number of classes, alphabetically.

        E.g.
            dir_path/class_x/xxx.ext
            dir_path/class_x/xxy.ext
            dir_path/class_x/xxz.ext
            dir_path/class_y/123.ext
            dir_path/class_y/nsdf3.ext
            dir_path/class_y/asd932_.ext

        Would produce two classes labeled 0 and 1 with 3 videos paths associated with each.

        Args:
            dir_path (str): Root directory to the video class directories .
        """
        assert g_pathmgr.exists(dir_path), f"{dir_path} not found."

        # Find all classes based on directory names. These classes are then sorted and indexed
        # from 0 to the number of classes.
        classes = sorted((f.name for f in pathlib.Path(dir_path).iterdir() if f.is_dir()))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        video_paths_and_label = make_dataset(dir_path, class_to_idx, extensions=("mp4", "avi"))
        assert len(video_paths_and_label) > 0, f"Failed to load dataset from {dir_path}."
        cls.labels = classes
        return cls(video_paths_and_label)

    @classmethod
    def from_directory_of_video_folders(cls, dir_path: str) -> LabeledVideoPaths:
        """
        Factory function that creates a LabeledVideoPaths object by parsing the structure
        of the given directory's subdirectories into the classification labels. It
        expects the directory format to be the following:
             dir_path/<class_name>/<video_name>/<frame_name>.jpg

        Classes are indexed from 0 to the number of classes, alphabetically.

        E.g.
            dir_path/class_x/vid1/xxx.ext
            dir_path/class_x/vid1/xxy.ext
            dir_path/class_x/vid2/xxz.ext
            dir_path/class_y/vid3/123.ext
            dir_path/class_y/vid4/nsdf3.ext
            dir_path/class_y/vid4/asd932_.ext

        Would produce two classes labeled 0 and 1 with 2 videos paths associated with each.

        Args:
            dir_path (str): Root directory to the video class directories .
        """
        assert g_pathmgr.exists(dir_path), f"{dir_path} not found."

        # Find all classes based on directory names. These classes are then sorted and indexed
        # from 0 to the number of classes.
        classes = sorted((f.name for f in pathlib.Path(dir_path).iterdir() if f.is_dir()))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        video_paths_and_label = make_dataset_from_video_folders(dir_path, class_to_idx, extensions=("jpg", "png"))
        assert len(video_paths_and_label) > 0, f"Failed to load dataset from {dir_path}."
        cls.labels = classes
        return cls(video_paths_and_label)


class LabeledVideoDataset(LabeledVideoDataset_):
    """
    LabeledVideoDataset handles the storage, loading, decoding and clip sampling for a
    video dataset. It assumes each video is stored as either an encoded video
    (e.g. mp4, avi) or a frame video (e.g. a folder of jpg, or png)
    """

    def __init__(
        self,
        labeled_video_paths: List[Tuple[str, Optional[dict]]],
        clip_sampler: ClipSampler,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[dict], Any]] = None,
        decode_audio: bool = True,
        decode_video: bool = True,
        decoder: str = "pyav",
        dataset_multiplier: int = 1,
    ):
        """
        Args:
            labeled_video_paths (List[Tuple[str, Optional[dict]]]): A list of pairs of
                video paths and optional labels.
            clip_sampler (ClipSampler): A clip sampler that samples a clip from a video.
            video_sampler (torch.utils.data.Sampler): A sampler that samples a video from the
                dataset.
            transform (Callable[[dict], Any]): A function that transforms a video dictionary
                into a tensor.
            decode_audio (bool): Whether to decode the audio of the video.
            decode_video (bool): Whether to decode the video of the video.
            decoder (str): The decoder to use for decoding the video.
            dataset_multiplier (int): The number of times to repeat the dataset.
        """

        self._decode_audio = decode_audio
        self._decode_video = decode_video
        self._transform = transform
        self._clip_sampler = clip_sampler
        self._labeled_videos = labeled_video_paths
        self._decoder = decoder

        # If a RandomSampler is used we need to pass in a custom random generator that
        # ensures all PyTorch multiprocess workers have the same random seed.
        self._video_random_generator = None
        if video_sampler == torch.utils.data.RandomSampler:
            self._video_random_generator = torch.Generator()
            self._video_sampler = video_sampler(
                self._labeled_videos,
                generator=self._video_random_generator,
                replacement=True,
                num_samples=len(self._labeled_videos) * dataset_multiplier,
            )
        else:
            self._video_sampler = video_sampler(self._labeled_videos)

        self._video_sampler_iter = None  # Initialized on first call to self.__next__()

        # Depending on the clip sampler type, we may want to sample multiple clips
        # from one video. In that case, we keep the store video, label and previous sampled
        # clip time in these variables.
        self._loaded_video_label = None
        self._loaded_clip = None
        self._last_clip_end_time = None
        self.video_path_handler = VideoPathHandler()

    @property
    def labels(self):
        """
        Returns:
            The list of class labels.
        """
        return self._labeled_videos.labels

    @property
    def videos_per_class(self):
        class_id_to_number = defaultdict(int)
        for _labeled_video in self._labeled_videos:
            label_info = _labeled_video[1]
            class_id_to_number[label_info["label"]] += 1
        class_ids = list(class_id_to_number.keys())
        return [class_id_to_number[class_id] for class_id in range(max(class_ids) + 1)]

    def __len__(self):
        if isinstance(self.video_sampler, torch.utils.data.SequentialSampler):
            return sum([1 for _ in self])
        elif isinstance(self.video_sampler, torch.utils.data.RandomSampler):
            return len(self.video_sampler)
        else:
            raise ValueError(f"Lenght calculation not implemented for sampler: {type(self.video_sampler)}.")


def labeled_video_dataset(
    data_path: str,
    clip_sampler: ClipSampler,
    video_sampler: Type[torch.utils.data.Sampler] = None,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    video_path_prefix: str = "",
    decode_audio: bool = True,
    decoder: str = "pyav",
    dataset_multiplier: int = 1,
) -> LabeledVideoDataset:
    """
    A helper function to create ``LabeledVideoDataset`` object for Ucf101 and Kinetics datasets.

    Args:
        data_path (str): Path to the data. The path type defines how the data
            should be read:

            * For a file path, the file is read and each line is parsed into a
              video path and label.
            * For a directory, the directory structure defines the classes
              (i.e. each subdirectory is a class).

        clip_sampler (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.

        video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.

        transform (Callable): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations to the clips. See the ``LabeledVideoDataset`` class for clip
                output format.

        video_path_prefix (str): Path to root directory with the videos that are
                loaded in ``LabeledVideoDataset``. All the video paths before loading
                are prefixed with this path.

        decode_audio (bool): If True, also decode audio from video.

        decoder (str): Defines what type of decoder used to decode a video.

    """
    labeled_video_paths = LabeledVideoPaths.from_path(data_path)
    labeled_video_paths.path_prefix = video_path_prefix
    video_sampler = torch.utils.data.RandomSampler(
        replacement=True, num_samples=len(labeled_video_paths) * dataset_multiplier
    )
    dataset = LabeledVideoDataset(
        labeled_video_paths,
        clip_sampler,
        video_sampler,
        transform,
        decode_audio=decode_audio,
        decoder=decoder,
    )
    return dataset


def download_ucf6(data_path: str):
    """
    Downloads the ucf6 dataset to the given path.
    """
    download_url = "https://github.com/fcakyon/video-transformers/releases/download/0.0.0/ucf6.zip"
    download_path = os.path.join(data_path, "ucf6.zip")
    download_file(download_url, download_path)
    with zipfile.ZipFile(download_path, "r") as zip_ref:
        zip_ref.extractall(data_path)
