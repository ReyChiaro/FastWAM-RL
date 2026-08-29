import json

from pathlib import Path
from typing import Any, TypeVar

import torch.nn as nn

from accelerate import init_empty_weights
from safetensors import safe_open


ModuleT = TypeVar("ModuleT", bound=nn.Module)


COMPONENT_DIRS = (
    "vae",
    "text_encoder",
    "video_expert",
    "action_expert",
    "proprio_encoder",
    "tokenizer",
    "scheduler",
)


def resolve_model_path(pretrained_model_name_or_path: str | Path) -> Path:
    """Resolve a local checkpoint bundle or a Hugging Face repository id."""

    path = Path(pretrained_model_name_or_path).expanduser()
    if path.exists():
        return path.resolve()

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=str(pretrained_model_name_or_path),
            allow_patterns=(
                "manifest.json",
                "weights/*.json",
                "weights/*.safetensors",
                "*.json",
                "*.safetensors",
                "tokenizer/*",
            ),
        )
    )


def is_per_module_bundle(model_path: str | Path) -> bool:
    """True when the bundle stores each component in its own subdirectory."""
    path = Path(model_path)
    if not path.is_dir():
        return False
    return any((path / name).is_dir() for name in COMPONENT_DIRS[:5]) and (
        path / "manifest.json"
    ).exists()


def component_dir(model_path: str | Path, component: str) -> Path | None:
    """Return the component subdirectory if it holds weights, else ``None``."""
    candidate = Path(model_path) / component
    if candidate.is_dir() and (
        any(candidate.glob("*.safetensors")) or any(candidate.glob("*.safetensors.index.json"))
    ):
        return candidate
    return None


def load_component_dir(module: ModuleT, component_dir: str | Path, strict: bool = True) -> ModuleT:
    """Load an unprefixed state dict from one component subdirectory."""
    component_dir = Path(component_dir)
    indexes = sorted(component_dir.glob("*.safetensors.index.json"))
    if indexes:
        if len(indexes) > 1:
            raise ValueError(f"Multiple safetensors indexes found under {component_dir}: {indexes}")
        index = json.loads(indexes[0].read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"Invalid safetensors index: {indexes[0]}")
        state_dict: dict[str, Any] = {}
        for shard_name in sorted(set(weight_map.values())):
            with safe_open(component_dir / shard_name, framework="pt", device="cpu") as handle:
                for key in weight_map:
                    if weight_map[key] == shard_name:
                        state_dict[key] = handle.get_tensor(key)
    else:
        files = sorted(component_dir.glob("*.safetensors"))
        if len(files) != 1:
            raise FileNotFoundError(
                f"Expected one safetensors file or one index under {component_dir}, "
                f"found {len(files)} files."
            )
        with safe_open(files[0], framework="pt", device="cpu") as handle:
            state_dict = {key: handle.get_tensor(key) for key in handle.keys()}

    incompatible = module.load_state_dict(state_dict, strict=strict, assign=True)
    if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        raise RuntimeError(
            f"Strict load failed for {component_dir}: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}."
        )
    return module


def construct_module(
    module_class: type[ModuleT],
    pretrained_model_name_or_path: str | Path | None,
    subfolder: str,
    init_kwargs: dict[str, Any],
    strict: bool = True,
) -> ModuleT:
    """Build a component from ``init_kwargs`` and optionally load its weights.

    Used by every submodel's ``from_pretrained``: allocate parameters on the
    meta device (never materialize random multi-billion tensors), then swap in
    the checkpoint tensors via ``assign=True``.  ``pretrained_model_name_or_path
    = None`` returns a randomly initialized module.
    """
    if pretrained_model_name_or_path is None:
        return module_class(**init_kwargs)
    with init_empty_weights(include_buffers=False):
        module = module_class(**init_kwargs)
    directory = component_dir(
        resolve_model_path(pretrained_model_name_or_path), subfolder
    )
    if directory is None:
        raise FileNotFoundError(
            f"No weights for component {subfolder!r} under "
            f"{pretrained_model_name_or_path}."
        )
    return load_component_dir(module, directory, strict=strict)
