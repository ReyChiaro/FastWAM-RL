import torch.nn as nn

from accelerate import Accelerator
from torch.utils.data import DataLoader

from evaluation.evaluator import Evaluator
from trainers.sft.trainer import SFTTrainer


class FlowGRPOTrainer(SFTTrainer):
    r"""
    Based on `SFTTrainer`.
    - Support Flow-GRPO: https://github.com/yifan123/flow_grpo
    """

    _trainer_ = "flow_grpo"

    def sampling_group(self):
        pass

    def update_actor(self):
        pass

    def train(
        self,
        accelerator: Accelerator,
        reference: nn.Module,
        actor: nn.Module,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        evaluator: Evaluator,
    ):
        pass
