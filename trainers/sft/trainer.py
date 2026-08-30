from trainers.base import BaseTrainer


class SFTTrainer(BaseTrainer):
    r"""
    Trainer for SFT.
    - LoRA of Full parameter training: providing utils for model parameter activation
        and deactivation and checkpoints management.
    """

    _trainer_ = "sft"

    pass
