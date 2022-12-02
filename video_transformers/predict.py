from collections import defaultdict
from typing import List

import pytorchvideo.data
import torch
import torch.utils.data
from iopath.common.file_io import g_pathmgr

from video_transformers.data import VideoPreprocessor
from video_transformers.pytorchvideo_wrapper.data.labeled_video_paths import LabeledVideoDataset, LabeledVideoPaths


def predict(
    model,
    video_or_folder_path,
    preprocessor_config: dict,
    labels: List[str],
    mode: str = "first_clip",
    device: str = None,
):
    """
    Predict the labels of a video or folder of videos.
    Supports local file path/folder directory, S3 URI and https URL.

    Args:
        model: The model to use for prediction.
        video_or_folder_path: The path to the video or folder of videos.
            Supports local file path/folder directory, S3 URI and https URL.
        preprocessor_config: The preprocessor config to use for prediction.
        labels: The labels to use for prediction.
        mode: The mode to use for prediction. Can be "first_clip", "average_all", "random_clip", "uniform_clip".
        device: The device to use for prediction. Can be "cpu" or "cuda:0".

    """
    labeled_video_paths = LabeledVideoPaths.from_path(video_or_folder_path)
    labeled_video_paths.path_prefix = ""
    video_sampler = torch.utils.data.SequentialSampler

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device

    if mode == "first_clip":
        clip_sampler = pytorchvideo.data.clip_sampling.UniformClipSamplerTruncateFromStart(
            clip_duration=preprocessor_config["clip_duration"],
            truncation_duration=preprocessor_config["clip_duration"],
        )
    elif mode == "average_all":
        clip_sampler = pytorchvideo.data.make_clip_sampler("uniform", preprocessor_config["clip_duration"])
    elif mode == "random_clip":
        clip_sampler = pytorchvideo.data.make_clip_sampler("random", preprocessor_config["clip_duration"])
    elif mode == "uniform_clip":
        clip_sampler = pytorchvideo.data.make_clip_sampler(
            "constant_clips_per_video", preprocessor_config["clip_duration"], 1
        )
    else:
        raise ValueError(
            f"Unknown mode: {mode}. Should be one of 'first_clip', 'average_all', 'random_clip', 'uniform_clip'."
        )

    preprocessor = VideoPreprocessor(**preprocessor_config)
    dataset = LabeledVideoDataset(
        labeled_video_paths,
        clip_sampler,
        video_sampler,
        preprocessor.val_transform,
        decode_audio=False,
        decoder="pyav",
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, drop_last=False)

    results = []
    model = model.to(device)
    model.eval()
    video_index_to_clip_predictions = defaultdict(list)
    video_index_to_video_name = {}
    with torch.inference_mode():
        for batch in dataloader:
            # parse batch
            video_index = batch["video_index"].tolist()[0]
            video_index_to_video_name[video_index] = batch["video_name"][0]
            batch["video"] = batch["video"].to(device)

            # predict
            output = model(batch["video"])

            # apply output activation function
            if model.task == "single_label_classification":
                probabilities = torch.nn.functional.softmax(output, dim=1)
            elif model.task == "multi_label_classification":
                probabilities = torch.nn.functional.sigmoid(output)
            else:
                raise ValueError(f"Unknown task: {model.task}")

            # save predictions
            video_index_to_clip_predictions[video_index].append(probabilities)

    # prepare results
    for video_index, clip_predictions in video_index_to_clip_predictions.items():
        video_name = video_index_to_video_name[video_index]
        probabilities = torch.mean(torch.stack(clip_predictions), dim=1)
        video_predictions = zip(labels, probabilities.tolist()[0])
        # sort predictions by probability
        video_predictions = dict(sorted(dict(video_predictions).items(), key=lambda item: item[1], reverse=True))
        # add video name
        video_results = {"filename": video_name, "predictions": video_predictions}
        results.append(video_results)

    if g_pathmgr.isfile(video_or_folder_path):
        results = results[0]

    return results
