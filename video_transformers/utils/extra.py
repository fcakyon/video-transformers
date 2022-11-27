from typing import Any, Tuple


def class_to_config(
    class_,
    allowed_types: Tuple[Any] = (int, float, str, dict, list, tuple, bool),
    ignored_attrs: Tuple[str] = ("config", "dump_patches", "training"),  # ignore nn.Module attributes
):
    """
    Converts a class attributes into a config dict.

    Args:
        class_: The class to convert.
        allowed_types: The attribute value types that are allowed in the config.
        ignored_attrs: The attributes that are ignored.

    Returns:
        The config dict.
    """
    config = {"name": class_.__class__.__name__}
    for attribute in dir(class_):
        if attribute not in ignored_attrs:
            if not attribute.startswith("__") and not attribute.startswith("_"):
                value = getattr(class_, attribute)
                if type(value) in allowed_types:
                    if type(value) == tuple:
                        value = list(value)
                    config[attribute] = value
    return config


def scheduler_to_config(scheduler):
    """
    Converts a scheduler and optimizer attributes into a config dict.

    Args:
        scheduler: The scheduler to convert.

    Returns:
        The config dict.
    """

    import torch

    if not isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR) or not len(scheduler._schedulers) == 2:
        raise TypeError("Scheduler must be a SequentialLR with 2 sub-schedulers")

    warmup_milestone_epoch = scheduler._milestones[0]
    warmup_scheduler = scheduler._schedulers[0]
    main_scheduler = scheduler._schedulers[1]
    if isinstance(main_scheduler, torch.optim.lr_scheduler.LinearLR):
        return {
            "optimizer": {
                "name": main_scheduler.optimizer.__class__.__name__,
                "defaults": main_scheduler.optimizer.defaults,
            },
            "warmup_scheduler": {
                "class": "torch.optim.lr_scheduler.LinearLR",
                "start_factor": warmup_scheduler.start_factor,
                "end_factor": warmup_scheduler.end_factor,
                "total_iters": warmup_scheduler.total_iters,
            },
            "main_scheduler": {
                "class": "torch.optim.lr_scheduler.LinearLR",
                "start_factor": main_scheduler.start_factor,
                "end_factor": main_scheduler.end_factor,
                "total_iters": main_scheduler.total_iters,
            },
            "warmup_milestone_epoch": warmup_milestone_epoch,
        }
    elif isinstance(main_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
        return {
            "optimizer": {
                "name": scheduler.optimizer.__class__.__name__,
                "defaults": scheduler.optimizer.defaults,
            },
            "warmup_scheduler": {
                "class": "torch.optim.lr_scheduler.LinearLR",
                "start_factor": warmup_scheduler.start_factor,
                "end_factor": warmup_scheduler.end_factor,
                "total_iters": warmup_scheduler.total_iters,
            },
            "main_scheduler": {
                "class": "torch.optim.lr_scheduler.CosineAnnealingLR",
                "T_max": main_scheduler.T_max,
            },
            "warmup_milestone_epoch": warmup_milestone_epoch,
        }
    elif isinstance(main_scheduler, torch.optim.lr_scheduler.MultiStepLR):
        return {
            "optimizer": {
                "name": scheduler.optimizer.__class__.__name__,
                "defaults": scheduler.optimizer.defaults,
            },
            "warmup_scheduler": {
                "class": "torch.optim.lr_scheduler.LinearLR",
                "start_factor": warmup_scheduler.start_factor,
                "end_factor": warmup_scheduler.end_factor,
                "total_iters": warmup_scheduler.total_iters,
            },
            "main_scheduler": {
                "class": "torch.optim.lr_scheduler.MultiStepLR",
                "milestones": main_scheduler.milestones,
                "gamma": main_scheduler.gamma,
            },
            "warmup_milestone_epoch": warmup_milestone_epoch,
        }
    else:
        raise TypeError(f"Scheduler not implemented {main_scheduler}")
