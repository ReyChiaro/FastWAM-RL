from typing import Any
from torch.utils.data import default_collate

from data.datasets.base import BaseDataset

PROMPT_TEMPLATE = "A video recorded from a robot's point of view executing the following instruction: {task}"


def libero_plus_collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return default_collate(samples)


class LiberoPlusDataset(BaseDataset):

    _keys_ = ["prompt", "video", "action", "proprio_states", "video_pad", "action_pad"]
