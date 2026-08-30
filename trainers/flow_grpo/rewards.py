import torch

import torch.nn as nn

# Define some reward methods here
# def action_score(...) -> ...


class RewardModel(nn.Module):
    r"""
    A frozen callable reward model for judging the quality of actor outputs.
    """

    def forward(self, actor_outputs) -> torch.Tensor:
        pass
