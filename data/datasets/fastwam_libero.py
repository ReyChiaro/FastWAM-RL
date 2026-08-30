import json
import numpy as np

from collections import defaultdict
from pathlib import Path
from typing import Any
from torch.utils.data import default_collate
from typing import Literal
from dataclasses import dataclass, field

from data.datasets.base import BaseDataset

PROMPT_TEMPLATE = "A video recorded from a robot's point of view executing the following instruction: {task}"


@dataclass
class FastWAMLiberoSplitMeta:
    r"""Meta info for each dataset split."""

    path: Path
    num_chunks: int
    num_episodes: int
    fps: int


@dataclass
class FastWAMLiberoEpisode:
    r"""
    A complete episode of one task, including videos in different cameras, action sequences and states.
    """

    split_name: str
    video_paths: dict[str, Path]
    state_path: Path
    num_frames: int
    task_instruct: str

    # Lazy load
    task_index: int = field(init=False)
    camera_videos: dict[str, np.ndarray] = field(init=False)
    actions: np.ndarray = field(init=False)
    proprio_states: np.ndarray = field(init=False)


# Register collate function with an identity name
# @collate_register("fastwam_libero")
def fastwam_libero_collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return default_collate(samples)


class FastWAMLiberoDataset(BaseDataset):
    r"""
    Dataset source: https://huggingface.co/datasets/yuanty/LIBERO-fastwam, a pre-processed LIBERO
        dataset with `libero_10, libero_goal, libero_object, libero_spatial` splits (the postfix
        is deleted). This dataset based on LeRobot v2.1.

    Four dataset splits are provided, each subfolder contains:
    - meta: Json/jsonl format, record meta info
        - task: Mappings between task_index and task instruction
        - info: Basic information of this split, including frames count, number of videos,
            video fps, data and video path, feature keys and shapes, etc.
        - episodes: List of episode info, including episode_index, task instruction and episode length
        - episode_stats: Statistics of each episode
    - data: Parquet format, stores actions, states and task_index
    - videos: MP4 videos, recommand use `torchcodec` to encode/decode
    """

    _keys_ = ["prompt", "video", "action", "proprio_states", "video_pad", "action_pad"]

    video_prefix = "observations.images"

    def __init__(
        self,
        root: str,
        splits: list[Literal["libero_10", "libero_goal", "libero_object", "libero_spatial"]],
        camera_keys: list[str] = ["image", "wrist_image"],
    ):
        super().__init__()

        root = Path(root)
        splits = set(splits)

        self.splits = splits
        self.camera_keys = camera_keys
        self.split_meta_info: dict[str, FastWAMLiberoSplitMeta] = {}
        self.episodes: list[FastWAMLiberoEpisode] = []

        for split in self.splits:
            split_path = root / split
            with open(split_path / "meta" / "info.json") as f:
                info = json.load(f)
                self.split_meta_info[split] = FastWAMLiberoSplitMeta(
                    path=split_path,
                    num_chunks=info.get("total_chunks"),
                    num_episodes=info.get("total_episodes"),
                    fps=info.get("fps"),
                )

            episode_map = {}
            with open(split_path / "meta" / "episodes.jsonl") as f:
                for line in f:
                    mapping = json.load(line)
                    episode_map[mapping.get("episode_index")] = {
                        "task": mapping.get("task")[0],
                        "length": mapping.get("length"),
                    }

            task_map = {}
            with open(split_path / "meta" / "tasks.jsonl") as f:
                for line in f:
                    mapping = json.load(line)
                    task_map[mapping.get("task_index")] = mapping.get("task")

            meta = self.split_meta_info[split]
            for c in range(meta.num_chunks):
                chunk_name = f"chunk-{c:03d}"
                for e in range(meta.num_episodes):
                    episode_name = f"episode-{e:06d}"
                    self.episodes.append(
                        FastWAMLiberoEpisode(
                            split_name=split,
                            task_instruct=episode_map[e]["task"],
                            num_frames=episode_map[e]["length"],
                            video_paths={
                                ck: split_path / "video" / chunk_name / ck / f"{episode_name}.mp4"
                                for ck in self.camera_keys
                            },
                            state_path=split_path / "data" / chunk_name / f"{episode_name}.parquet",
                        )
                    )

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, frame_index: int) -> dict[str, Any]:
        pass
