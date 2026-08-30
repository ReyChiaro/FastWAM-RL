import torch.nn as nn

from accelerate import Accelerator
from torch.utils.data import DataLoader

from typing import Literal, Optional
from omegaconf import OmegaConf

from evaluation.evaluator import Evaluator


class BaseTrainer:
    r"""
    Trainer for basic functions defined in training loop.
    - Project initialization
    - Parallel training on multi-node and multi-gpu
    - Checkpoint management
    - Logging
    """

    _trainer_: Literal["sft", "flow_grpo", "none"] = "none"  # "none" for base trainer, only testing interfaces.

    def __init__(self, project_cfgs: OmegaConf, train_cfgs: OmegaConf, optim_cfgs: OmegaConf):
        r"""
        Args:
            optim_cfgs: Hydra configs, including torch optimizer configs and optional lr_sheduler.
                keys: ["optimizer", "lr_scheduler"]
            train_cfgs: Hydra configs, including train steps and states managements.
                keys: [
                    "num_epochs", "max_training_steps",    # Controlling training states
                    "eval_step", "log_step", "save_step",  # Controlling logs
                ]
            project_cfgs: Hydra configs, including project names, folders and timestamps.
                keys: ["name", "output_dir", "log_dir", "checkpoint_dir", "timestamp"]
                NOTE: Shared among different processors to keep timestamp/folder consistency.
        """
        pass

    def init_project(self):
        r"""
        Create folders, set random seeds, set logger
        """

    def save_checkpoints(self):
        r"""
        Save model checkpoints and optional optimizer checkpoints, supporting parallel.
        """
        pass

    def load_checkpoints(self):
        pass

    def train(
        self,
        accelerator: Accelerator,
        model: nn.Module,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        evaluator: Optional[Evaluator] = None,
    ):
        pass
