import torch.nn as nn


class CheckpointModule:
    r"""
    Handle checkpoints loading and saving.
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
    ) -> nn.Module:
        pass
