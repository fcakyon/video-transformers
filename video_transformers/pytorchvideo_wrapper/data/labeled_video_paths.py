# modified from https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/data/labeled_video_paths.py
# and https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/data/labeled_video_dataset.py

from __future__ import annotations

import logging
import os
import pathlib
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import torch
from iopath.common.file_io import g_pathmgr
from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data.labeled_video_dataset import LabeledVideoDataset as LabeledVideoDataset_
from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths as LabeledVideoPaths_
from pytorchvideo.data.video import VideoPathHandler
from torchvision.datasets.folder import find_classes, has_file_allowed_extension, make_dataset

logger = logging.getLogger(__name__)


def make_dataset_from_video_folders(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_folder(x: str) -> bool:
            if g_pathmgr.ls(x):
                return has_file_allowed_extension(g_pathmgr.ls(x)[0], extensions)
            else:
                return False

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, fnames, _ in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_folder(path):
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances


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

        if g_pathmgr.isfile(data_path) and data_path.endswith(".csv"):
            return LabeledVideoPaths.from_csv(data_path)
        elif g_pathmgr.isfile(data_path) and has_file_allowed_extension(data_path, extensions=("mp4", "avi")):
            return LabeledVideoPaths.from_video_path(data_path)
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
    def from_video_path(cls, video_path: str) -> LabeledVideoPaths:
        """
        Creates a LabeledVideoPaths object from a single video path.
        Args:
            video_path (str): The path to the video.
        """
        return LabeledVideoPaths([(video_path, -1)])

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

        self._len = None

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
        if self._len is not None:
            return self._len

        if isinstance(self.video_sampler, torch.utils.data.SequentialSampler):
            # self._len = sum([1 for _ in self])
            self._len = len(self.video_sampler)
            return self._len
        elif isinstance(self.video_sampler, torch.utils.data.RandomSampler):
            self._len = len(self.video_sampler)
            return self._len
        else:
            raise ValueError(f"Length calculation not implemented for sampler: {type(self.video_sampler)}.")
