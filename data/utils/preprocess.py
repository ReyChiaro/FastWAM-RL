import torch
import pyarrow
import numpy as np

from typing import Optional
from pathlib import Path
from torchcodec.decoders import VideoDecoder
from pyarrow import parquet


def decode_video(mp4_path: Path, fps: float, frame_indices: Optional[list[int]] = None) -> torch.Tensor:
    # Use ffmpeg
    decoder = VideoDecoder(mp4_path, device="cpu", seek_mode="approximate")
    if frame_indices is None:
        frames = decoder.get_all_frames(fps=fps).data
    else:
        frames = decoder.get_frames_at(frame_indices).data
    return frames.to(dtype=torch.float32)


def load_parquet(parquet_path: Path, columns: Optional[list[str]]) -> dict[str, torch.Tensor]:
    table: pyarrow.Table = parquet.read_table(parquet_path, columns=columns)
    column_tensors = {}
    for name in columns:
        array = table[name].combine_chunks().to_numpy(zero_copy_only=False)
        if array.dtype == object:
            array = np.stack(array)
        column_tensors[name] = torch.from_numpy(np.asarray(array).copy()).to(dtype=torch.float32)
    return column_tensors
