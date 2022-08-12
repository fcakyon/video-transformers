from typing import List

from torch import nn


def unfreeze_last_n_stages(stages: List[nn.Module], n: int):
    if n == -1:  # dont freeze if -1
        return

    num_stages = len(stages)
    num_stages_to_freeze = num_stages - n
    for i, stage in enumerate(stages):
        if i >= num_stages_to_freeze:
            break
        for param in stage.parameters():
            param.requires_grad = False


def get_num_trainable_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_num_total_params(model: nn.Module):
    return sum(param.numel() for param in model.parameters())
