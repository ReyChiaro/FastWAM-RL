import json
import torch
import numpy as np
import torchvision.transforms.functional as T

from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from torch.utils.data import default_collate

from data.datasets.base import BaseDataset
from data.utils.preprocess import decode_video, load_parquet

PROMPT_TEMPLATE = "A video recorded from a robot's point of view executing the following instruction: {task}"


@dataclass
class DataStats:

    minimum: np.ndarray
    maximum: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray
    q99: np.ndarray

    @classmethod
    def from_path(cls, path: str, key: str) -> "DataStats":
        with open(path) as f:
            stats = json.load(f).get(key)["default"]
        return cls(
            minimum=np.array(stats["global_min"], dtype=np.float32),
            maximum=np.array(stats["global_max"], dtype=np.float32),
            mean=np.array(stats["global_mean"], dtype=np.float32),
            std=np.array(stats["global_std"], dtype=np.float32),
            q01=np.array(stats["global_q01"], dtype=np.float32),
            q99=np.array(stats["global_q99"], dtype=np.float32),
        )


@dataclass
class FastWAMLiberoSplitMeta:
    r"""Meta info for each dataset split."""

    path: Path
    num_chunks: int
    num_episodes: int
    fps: int

    @classmethod
    def from_path(cls, path: str) -> "FastWAMLiberoSplitMeta":
        with open(path / "meta" / "info.json") as f:
            info = json.load(f)
        return FastWAMLiberoSplitMeta(
            path=path,
            num_chunks=info.get("total_chunks"),
            num_episodes=info.get("total_episodes"),
            fps=info.get("fps"),
        )


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
        dataset_stats: str,
        video_size: tuple[int] = [224, 224],
        action_horizon: int = 32,
        video_stride: int = 4,
        camera_keys: list[str] = ["image", "wrist_image"],
        action_key: str = "action",
        proprio_key: str = "observation.state",
    ):
        r"""
        Args:
            action_horizon (int): Num of action frames in one window, default 32
            video_stride (int): Video frames sampling frequence, default 4
            dataset_stats (str): The statistics of datasets for all given splits, should be pre-computed.
        """
        super().__init__()

        root = Path(root)
        splits = set(splits)

        self.splits = splits
        self.video_size = video_size
        self.action_horizon = action_horizon
        self.video_stride = video_stride
        self.camera_keys = camera_keys
        self.action_key = action_key
        self.proprio_key = proprio_key

        self.split_meta_info: dict[str, FastWAMLiberoSplitMeta] = {}
        self.episodes: list[FastWAMLiberoEpisode] = []

        self.action_stats = DataStats.from_path(dataset_stats, "action")
        self.proprio_stats = DataStats.from_path(dataset_stats, "state")

        for split in self.splits:
            split_path = root / split
            self.split_meta_info[split] = FastWAMLiberoSplitMeta.from_path(split_path)

            episode_map = {}
            with open(split_path / "meta" / "episodes.jsonl") as f:
                for line in f:
                    mapping = json.loads(line)
                    episode_map[mapping.get("episode_index")] = {
                        "task": mapping.get("tasks")[0],
                        "length": mapping.get("length"),
                    }

            task_map = {}
            with open(split_path / "meta" / "tasks.jsonl") as f:
                for line in f:
                    mapping = json.loads(line)
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

        self._episode_end_indices = np.cumsum([e.num_frames for e in self.episodes]).tolist()

    def __len__(self) -> int:
        return len(self.episodes)

    def concat_videos(self, videos: list[torch.Tensor]) -> torch.Tensor:
        videos = [
            T.resize(v, self.video_size, interpolation=T.InterpolationMode.BICUBIC).clamp(0.0, 1.0) for v in videos
        ]
        return torch.cat(videos, dim=-1)

    def normalize(self, values: torch.Tensor, stats: DataStats) -> torch.Tensor:
        minimum = torch.from_numpy(stats.minimum).to(values)
        maximum = torch.from_numpy(stats.maximum).to(values)
        value_range = maximum - minimum
        scale = torch.where(value_range < 1e-4, torch.ones_like(value_range), 2 / value_range)
        offset = torch.where(value_range < 1e-4, -minimum, -1 - scale * minimum)
        return (values * scale + offset).clamp(-5, 5)

    def denormalize(self, values: torch.Tensor, stats: DataStats) -> torch.Tensor:
        minimum = torch.from_numpy(stats.minimum).to(values)
        maximum = torch.from_numpy(stats.maximum).to(values)
        value_range = maximum - minimum
        scale = torch.where(value_range < 1e-4, torch.ones_like(value_range), 2 / value_range)
        offset = torch.where(value_range < 1e-4, -minimum, -1 - scale * minimum)
        return (values - offset) / scale

    def __getitem__(self, frame_index: int) -> dict[str, Any]:
        # Find the corresponding episode
        episode_index = bisect_right(self._episode_end_indices, frame_index)
        frame_start = 0 if episode_index == 0 else self._episode_end_indices[episode_index - 1]
        episode = self.episodes[episode_index]

        offset = frame_index - frame_start
        num_frames = episode.num_frames

        # Config frames/indices to load and which are padding
        action_indices = offset + list(range(self.action_horizon))
        video_indices = offset + list(range(0, self.action_horizon + 1, self.video_stride))
        action_pad = action_indices >= num_frames
        video_pad = video_indices >= num_frames

        # Load videos, actions and states
        # Videos from different cameras are concatenated horizontally and sequentially
        videos = self.concat_videos([decode_video(episode.video_paths[k]) for k in self.camera_keys])
        states = load_parquet(episode.state_path, columns=[self.action_key, self.proprio_key])

        # Preprocess
        prompt = PROMPT_TEMPLATE.format_map({"task": episode.task_instruct})
        actions = self.normalize(states[self.action_key], self.action_stats)
        # Only first proprio state in the horizon is used
        proprios = self.normalize(states[self.proprio_stats][0], self.proprio_stats)

        # Note that the returned dict have the same keys defined in self._keys_
        return {
            "prompt": prompt,
            "video": videos,
            "action": actions,
            "proprio_states": proprios,
            "video_pad": video_pad,
            "action_pad": action_pad,
        }
