import torch


def get_multistep_scheduler_with_warmup(
    optimizer: torch.optim.Optimizer, max_epochs: int = 12, warmup_epochs: float = 0.1
):
    """
    Torch multistep learning rate scheduler with warmup.
    Decrease the learning rate at milestones by a factor of 0.1.
    Milestones are chosen as 7/10 and 9/10 of total epochs.
    """
    warmup_milestone_epoch = max(1.0, int(warmup_epochs * max_epochs))
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=max(1.0, warmup_milestone_epoch),
    )

    ms_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            round((max_epochs - 1) * 7 / 10),
            round((max_epochs - 1) * 9 / 10),
        ],
        gamma=0.1,
        last_epoch=-1,
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [warmup_scheduler, ms_scheduler],
        milestones=[warmup_milestone_epoch],
    )

    return scheduler


def get_linear_scheduler_with_warmup(
    optimizer: torch.optim.Optimizer, max_epochs: int = 12, warmup_epochs: float = 0.1
):
    """
    Torch linear learning rate scheduler with warmup.
    """
    warmup_milestone_epoch = max(1.0, int(warmup_epochs * max_epochs))
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_milestone_epoch,
    )

    linear_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.01,
        total_iters=max_epochs - warmup_milestone_epoch,
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [warmup_scheduler, linear_scheduler],
        milestones=[warmup_milestone_epoch],
    )

    return scheduler


def get_cosineannealing_scheduler_with_warmup(
    optimizer: torch.optim.Optimizer, max_epochs: int = 12, warmup_epochs: float = 0.1
):

    warmup_milestone_epoch = max(1.0, int(warmup_epochs * max_epochs))
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=max(1.0, warmup_milestone_epoch),
    )

    cosineannealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs - warmup_milestone_epoch,
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [warmup_scheduler, cosineannealing_scheduler],
        milestones=[warmup_milestone_epoch],
    )

    return scheduler
