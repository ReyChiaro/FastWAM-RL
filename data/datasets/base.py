from abc import ABC

from torch.utils.data import Dataset, default_collate
from typing import Any, Callable

# Collate backend functions are registered using decorators at importing time.
collate_backends = {}


# When configured with hydra, use _target_ key to specify the collate_fn
def build_collate_fn(backend_name: str) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    r"""
    Collate function handle data pre-processing, normalization and data batching.
    """
    if backend_name in collate_backends:
        return collate_backends[backend_name]
    print(f"[WARNING] No collate function is configured, use the default.")
    return default_collate


class BaseDataset(ABC, Dataset):
    r"""
    Dataset operates as a storage reader from medias.
    BaseDataset provide public shared interfaces.
    """

    # Defines the valid keys of one dict item.
    _keys_: list[str] = []

    def __init__(self):
        super().__init__()
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self) -> dict[str, Any]:
        pass
