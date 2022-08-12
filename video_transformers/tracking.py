import os
from typing import Optional, Union

from accelerate.logging import get_logger
from accelerate.tracking import GeneralTracker, is_tensorboard_available

from video_transformers.utils.imports import check_requirements, is_layer_available, is_neptune_available
from video_transformers.utils.logger import _flatten_dict, _sanitize_params

logger = get_logger(__name__)

if is_neptune_available():
    import neptune.new as neptune

if is_layer_available():
    import layer
    import layer.contracts.projects

if is_tensorboard_available():
    from accelerate.tracking import tensorboard

# TODO: log image (confusion mat)


class NeptuneTracker(GeneralTracker):
    """
    A `Tracker` class that supports `neptune`. Should be initialized at the start of your script.
    Args:
        run_name (`str`):
            The name of the experiment run.
        kwargs:
            Additional key word arguments passed along to the `neptune.init` method.
    """

    name = "neptune"
    requires_logging_directory = False

    def __init__(self, run_name: str, **kwargs):
        check_requirements(["neptune-client"])

        self.run_name = run_name

        self.run: neptune.Run = neptune.init(name=self.run_name, **kwargs)
        logger.info(f"Initialized Neptune project {self.run_name}")
        logger.info("Make sure to log any initial configurations with `self.store_init_configuration` before training!")

    @property
    def tracker(self):
        return self.run

    def store_init_configuration(self, values: dict):
        """
        Logs `values` as hyperparameters for the run. Should be run at the beginning of your experiment.
        Args:
            values (Dictionary `str` to `bool`, `str`, `float` or `int`):
                Values to be stored as initial hyperparameters as key-value pairs. The values need to have type `bool`,
                `str`, `float`, `int`, or `None`.
        """
        self.run["config"] = values
        logger.info("Stored initial configuration hyperparameters to Neptune")

    def log(self, values: dict, step: Optional[int] = None, **kwargs):
        """
        Logs `values` to the current run.
        Args:
            values (Dictionary `str` to `str`, `float`, `int` or `dict` of `str` to `float`/`int`):
                Values to be logged as key-value pairs. The values need to have type `str`, `float`, `int` or `dict` of
                `str` to `float`/`int`.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs:
                Additional key word arguments passed along to the `neptune.log` method.
        """
        for key, value in values.items():
            if isinstance(value, (dict, int, float, bool, str)):
                self.run[key].log(value, step=step, **kwargs)
            else:
                raise NotImplementedError(f"NeptuneTracker does not support logging {type(value)}")
        logger.info("Successfully logged to Neptune")

    def finish(self):
        """
        Closes `neptune` writer
        """
        self.run.stop()
        logger.info("Neptune run closed")


class TensorBoardTracker(GeneralTracker):
    """
    A `Tracker` class that supports `tensorboard`. Should be initialized at the start of your script.

    Args:
        run_name (`str`):
            The name of the experiment run
        logging_dir (`str`, `os.PathLike`):
            Location for TensorBoard logs to be stored.
    """

    name = "tensorboard"
    requires_logging_directory = True

    def __init__(self, run_name: str, logging_dir: Optional[Union[str, os.PathLike]]):
        self.run_name = run_name
        self.logging_dir = os.path.join(logging_dir)
        self.writer = tensorboard.SummaryWriter(self.logging_dir)
        logger.info(f"Initialized TensorBoard project {self.run_name} logging to {self.logging_dir}")
        logger.info("Make sure to log any initial configurations with `self.store_init_configuration` before training!")

    @property
    def tracker(self):
        return self.writer

    def store_init_configuration(self, values: dict):
        """
        Logs `values` as hyperparameters for the run. Should be run at the beginning of your experiment.

        Args:
            values (Dictionary `str` to `bool`, `str`, `float` or `int`):
                Values to be stored as initial hyperparameters as key-value pairs. The values need to have type `bool`,
                `str`, `float`, `int`, or `None`.
        """
        values = _flatten_dict(values)
        values = _sanitize_params(values)
        self.writer.add_hparams(values, metric_dict={})
        self.writer.flush()
        logger.info("Stored initial configuration hyperparameters to TensorBoard")

    def log(self, values: dict, step: Optional[int] = None):
        """
        Logs `values` to the current run.

        Args:
            values (Dictionary `str` to `str`, `float`, `int` or `dict` of `str` to `float`/`int`):
                Values to be logged as key-value pairs. The values need to have type `str`, `float`, `int` or `dict` of
                `str` to `float`/`int`.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
        """
        for k, v in values.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, global_step=step)
            elif isinstance(v, str):
                self.writer.add_text(k, v, global_step=step)
            elif isinstance(v, dict):
                v = _flatten_dict(v)
                self.writer.add_scalars(k, v, global_step=step)
        self.writer.flush()
        logger.info("Successfully logged to TensorBoard")

    def finish(self):
        """
        Closes `TensorBoard` writer
        """
        self.writer.close()
        logger.info("TensorBoard writer closed")


class LayerTracker(GeneralTracker):
    """
    A `Tracker` class that supports `layer`. Should be initialized at the start of your script.
    Args:
        run_name (`str`):
            The name of the experiment run.
        kwargs:
            Additional key word arguments passed along to the `layer.init` method.
    """

    name = "layer"
    requires_logging_directory = False

    def __init__(self, run_name: str, **kwargs):
        check_requirements(["layer"])

        self.run_name = run_name
        self.model_name = kwargs.get("model_name", run_name)

        api_token = kwargs.get("api_token", None)
        kwargs.pop("api_token", None)
        if api_token is not None:
            layer.login_with_access_token(api_token)
        self.project: layer.contracts.projects.Project = layer.init(project_name=self.run_name, **kwargs)
        logger.info(f"Initialized Layer project {self.run_name}")
        logger.info("Make sure to log any initial configurations with `self.store_init_configuration` before training!")

    @property
    def tracker(self):
        return layer

    def store_init_configuration(self, values: dict):
        """
        Logs `values` as hyperparameters for the run. Should be run at the beginning of your experiment.
        Args:
            values (Dictionary `str` to `bool`, `str`, `float` or `int`):
                Values to be stored as initial hyperparameters as key-value pairs. The values need to have type `bool`,
                `str`, `float`, `int`, or `None`.
        """
        layer.log({"config": values})
        logger.info("Stored initial configuration hyperparameters to Layer")

    def log(self, values: dict, step: Optional[int] = None, **kwargs):
        """
        Logs `values` to the current run.
        Args:
            values (Dictionary `str` to `str`, `float`, `int` or `dict` of `str` to `float`/`int`):
                Values to be logged as key-value pairs. The values need to have type `str`, `float`, `int` or `dict` of
                `str` to `float`/`int`.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
            kwargs:
                Additional key word arguments passed along to the `layer.log` method.
        """
        layer.log(values, step=step, **kwargs)
        logger.info("Successfully logged to Layer")

    def finish(self):
        """
        Closes `Layer` writer
        """
        pass
        logger.info("Layer run closed")
