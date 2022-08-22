from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.tracking import GeneralTracker
from torch.optim.lr_scheduler import _LRScheduler
from tqdm.auto import tqdm

import video_transformers.data
from video_transformers.modeling import VideoModel
from video_transformers.schedulers import get_linear_scheduler_with_warmup
from video_transformers.tasks.single_label_classification import SingleLabelClassificationTaskMixin
from video_transformers.tracking import TensorBoardTracker
from video_transformers.utils.extra import scheduler_to_config
from video_transformers.utils.file import increment_path

logger = get_logger(__name__)


class BaseTrainer:
    def __init__(
        self,
        datamodule: video_transformers.data.VideoDataModule,
        model: VideoModel,
        max_epochs: int = 12,
        cpu: bool = False,
        mixed_precision: str = "no",
        output_dir: str = "runs",
        seed: int = 42,
        trackers: list[GeneralTracker] = None,
        checkpoint_save: bool = True,
        checkpoint_save_interval: int = 1,
        checkpoint_save_policy: str = "epoch",
        experiment_name="exp",
        optimizer: torch.optim.Optimizer = None,
        scheduler: _LRScheduler = None,
        config_dict: dict = None,
        loss_function: torch.nn.modules.loss._Loss = None,
    ):
        """
        Args:
            datamodule: A `VideoDataModule` instance.
            model: A `nn.Module` instance.
            max_epochs:
            cpu: Use CPU if True.
            mixed_precision: One of ['no', 'fp16', 'bf16'].
            output_dir: The directory to save outputs.
            lr: The max learning rate.
            num_epochs: The number of epochs.
            seed: The random seed.
            trackers: A list of `GeneralTracker` instances.
            checkpoint_save: Whether to save checkpoints.
            checkpoint_save_interval: The interval between saving checkpoints.
            checkpoint_save_policy: One of ['epoch', 'step'].
            experiment_name: The name of the experiment.
            optimizer: The optimizer.
            scheduler: The scheduler.
            config_dict: Optional config setting to be logged.
        """
        self.experiment_dir = increment_path(Path(output_dir) / experiment_name, exist_ok=False)

        self._set_optimizer(model, optimizer)
        self._set_scheduler(scheduler, max_epochs)
        self._set_loss_function(loss_function)
        self._set_trackers(trackers, experiment_name)

        self.accelerator = Accelerator(
            cpu=cpu, mixed_precision=mixed_precision, log_with=self.trackers, logging_dir=self.experiment_dir
        )

        labels = datamodule.labels

        # set haparams to be logged
        ignore_args = [
            "self",
            "ignore_args",
            "model",
            "datamodule",
            "output_dir",
            "experiment_name",
            "optimizer",
            "scheduler",
            "config_dict",
            "loss_function",
            "trackers",
            "vt_version",
        ]
        vt_version = {"video_transformers_version": video_transformers.__version__}
        hparams = locals()
        hparams = {k: v for k, v in hparams.items() if k not in ignore_args}
        if config_dict is not None:
            hparams.update(config_dict)
        if isinstance(model, VideoModel):
            hparams.update({"model": model.config})
        if isinstance(datamodule, video_transformers.data.VideoDataModule):
            hparams.update({"data": datamodule.config})
        if isinstance(self.scheduler, torch.optim.lr_scheduler.SequentialLR):
            hparams.update({"scheduler": scheduler_to_config(self.scheduler)})
        hparams.update(vt_version)
        self.hparams = hparams

        # We need to initialize the trackers we use, and also store our configuration
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(experiment_name, hparams)

        # Set the seed before splitting the data.
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.val_dataloader,
            self.test_dataloader,
            self.scheduler,
        ) = self.accelerator.prepare(
            model,
            self.optimizer,
            datamodule.train_dataloader,
            datamodule.val_dataloader,
            datamodule.test_dataloader,
            self.scheduler,
        )

        self.overall_step = 0
        self.overall_epoch = 0
        self.starting_epoch = 0

        self.labels = labels

        self.last_saved_checkpoint_epoch = -1

    @property
    def train_metrics(self):
        raise NotImplementedError()

    @property
    def val_metrics(self):
        raise NotImplementedError()

    @property
    def last_train_result(self):
        return None

    @property
    def last_val_result(self):
        return None

    def _set_optimizer(self, model, optimizer):
        if optimizer is None:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        self.optimizer = optimizer

    def _set_scheduler(self, scheduler, max_epochs=None):
        if scheduler is None:
            scheduler = get_linear_scheduler_with_warmup(self.optimizer, max_epochs=max_epochs)
        self.scheduler = scheduler

    def _set_loss_function(self, loss_function):
        if loss_function is None:
            loss_function = torch.nn.CrossEntropyLoss()
        self.loss_function = loss_function

    def _set_trackers(self, trackers, experiment_name: str):
        tensorboard_present = False
        tb_tracker = TensorBoardTracker(run_name=experiment_name, logging_dir=self.experiment_dir)
        if trackers is None:
            trackers = tb_tracker
            tensorboard_present = True
        else:
            for tracker in trackers:
                if tracker.name == "tensorbaord":
                    tracker = tb_tracker
                    tensorboard_present = True
        if not tensorboard_present:
            trackers.append(tb_tracker)
        self.trackers = trackers

    def _log_last_lr(self):
        if self.accelerator.is_main_process:
            lr = self.optimizer.param_groups[-1]["lr"]
            self.accelerator.log({"lr": lr}, step=self.overall_epoch)

    def _get_last_train_score(self) -> Union[None, Dict[str, float]]:
        if self.last_train_result is not None:
            first_train_metric = list(self.last_train_result.keys())[0]
            return ("train/" + first_train_metric, self.last_train_result[first_train_metric].mean())
        else:
            return None

    def _get_last_val_score(self) -> Union[None, Dict[str, float]]:
        if self.last_val_result is not None:
            first_val_metric = list(self.last_val_result.keys())[0]
            return ("val/" + first_val_metric, self.last_val_result[first_val_metric].mean())
        else:
            return None

    def _save_state_and_checkpoint(self):
        if self.accelerator.is_main_process:
            output_name = "checkpoint"
            save_path = Path(self.experiment_dir) / output_name
            if self.hparams["checkpoint_save_policy"] in ["steps", "step"]:
                if self.overall_step % self.hparams["checkpoint_save_interval"] == 0:
                    self.accelerator.save_state(save_path)
            elif self.hparams["checkpoint_save_policy"] in ["epochs", "epoch"]:
                if (
                    self.last_saved_checkpoint_epoch != self.overall_epoch
                    and self.overall_epoch % self.hparams["checkpoint_save_interval"] == 0
                ):
                    self.accelerator.save_state(save_path)
                    self.save_checkpoint(save_path)
                    self.last_saved_checkpoint_epoch = self.overall_epoch
            else:
                raise ValueError(f"Unknown checkpoint save policy: {self.hparams['checkpoint_save_policy']}")

    def save_checkpoint(self, save_path: Union[str, Path]):
        config = self.model.config.copy()
        data_config = self.hparams["data"]
        try:
            scheduler_config = scheduler_to_config(self.scheduler)
        except TypeError:
            scheduler_config = None
        config.update(
            {
                "preprocessor": {
                    "means": data_config["preprocessor_config"]["means"],
                    "stds": data_config["preprocessor_config"]["stds"],
                    "min_short_side": data_config["preprocessor_config"]["min_short_side"],
                    "input_size": data_config["preprocessor_config"]["input_size"],
                    "clip_duration": data_config["preprocessor_config"]["clip_duration"],
                    "num_timesteps": data_config["preprocessor_config"]["num_timesteps"],
                },
                "labels": data_config["labels"],
                "scheduler": scheduler_config,
            }
        )
        self.model.save_pretrained(save_path, config)

    def _update_state(self, loss):
        self.accelerator.backward(loss)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.overall_step += 1

    def _one_train_loop(self, pbar, len_train_dataloader):
        pbar.set_description(f"Epoch {self.overall_epoch} (Train)")

        self.model.train()
        train_loss = 0
        train_dataloader_iter = iter(self.train_dataloader)

        for step in range(len_train_dataloader):
            pbar.update(1)

            batch = next(train_dataloader_iter)
            loss = self.training_step(batch)
            train_loss += loss

            # update accelerator, optimizer, scheduler
            self._update_state(loss)

            # update pbar with train batch loss
            pbar.set_postfix({"loss": f"{loss:.4f}"})

            self._save_state_and_checkpoint()

        return train_loss

    def _one_val_loop(self, pbar, len_val_dataloader):
        pbar.set_description(f"Epoch {self.overall_epoch} (Val)  ")

        self.model.eval()
        val_loss = 0
        val_dataloader_iter = iter(self.val_dataloader)

        for step in range(len_val_dataloader):
            pbar.update(1)

            batch = next(val_dataloader_iter)
            loss = self.validation_step(batch)
            val_loss += loss

            # update pbar with val batch loss
            train_score = self._get_last_train_score()
            val_score = self._get_last_val_score()
            if val_score is not None:
                pbar.set_postfix(
                    {
                        "loss": f"{loss:.4f}",
                        train_score[0]: f"{train_score[1]:.3f}",
                    }
                )
            else:
                pbar.set_postfix(
                    {
                        "loss": f"{loss:.4f}",
                        train_score[0]: f"{train_score[1]:.3f}",
                    }
                )

        return val_loss

    def fit(self):
        self.accelerator.print(f"Trainable parameteres: {self.model.num_trainable_params}")
        self.accelerator.print(f"Total parameteres: {self.model.num_total_params}")

        len_train_dataloader = len(self.train_dataloader)
        len_val_dataloader = len(self.val_dataloader)

        for epoch in range(self.starting_epoch, self.hparams["max_epochs"]):
            self._log_last_lr()

            with tqdm(
                total=len_train_dataloader + len_val_dataloader,
                unit=" batch",
                disable=not self.accelerator.is_local_main_process,
            ) as pbar:

                # train loop
                train_loss = self._one_train_loop(pbar, len_train_dataloader)

                # call the end of train epoch hook
                self.on_training_epoch_end()

                # update pbar with train epoch loss and score
                train_loss = train_loss / len_train_dataloader
                train_score = self._get_last_train_score()
                pbar.set_postfix({"loss": f"{train_loss:.4f}", train_score[0]: f"{train_score[1]:.3f}"})

                # val loop
                with torch.inference_mode():
                    val_loss = self._one_val_loop(pbar, len_val_dataloader)

                # call the end of val epoch hook
                self.on_validation_epoch_end()

                # update pbar with val epoch loss and score
                val_loss = val_loss / len_val_dataloader
                val_score = self._get_last_val_score()
                pbar.set_postfix(
                    {
                        "loss": f"{val_loss:.4f}",
                        val_score[0]: f"{val_score[1]:.3f}",
                        train_score[0]: f"{train_score[1]:.3f}",
                    }
                )
                pbar.set_description(f"Epoch {self.overall_epoch} (Done) ")

            self._save_state_and_checkpoint()
            self.overall_epoch = int(self.overall_epoch + 1)
            self.scheduler.step()


class SingleLabelClassificationTrainer(SingleLabelClassificationTaskMixin, BaseTrainer):
    pass


def trainer_factory(task: str):
    if task == "single_label_classification":
        return SingleLabelClassificationTrainer
    else:
        raise ValueError(f"Unknown trainer task: {task}")
