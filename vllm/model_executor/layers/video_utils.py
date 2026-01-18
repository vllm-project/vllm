from __future__ import annotations

from typing import Iterable, List

import numpy as np
import torch
from PIL import Image


def pil_frames_to_uint8_tensor(frames: Iterable[Image.Image]) -> torch.Tensor:
    """Convert a sequence of PIL RGB frames to uint8 torch tensor [F, H, W, 3]."""
    arrs: List[np.ndarray] = []
    for img in frames:
        if img.mode != "RGB":
            img = img.convert("RGB")
        arrs.append(np.asarray(img, dtype=np.uint8))

    if not arrs:
        return torch.empty((0, 0, 0, 3), dtype=torch.uint8)

    stacked = np.stack(arrs, axis=0)  # [F, H, W, 3]
    return torch.from_numpy(stacked)


def uint8_tensor_to_pil_frames(video: torch.Tensor) -> List[Image.Image]:
    """Convert uint8 torch tensor [F, H, W, 3] to list of PIL RGB images."""
    if video.dtype != torch.uint8:
        raise ValueError("Expected uint8 tensor for conversion to PIL frames")
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError("Expected shape [F, H, W, 3] for video tensor")

    video_np = video.detach().cpu().numpy()
    frames = [Image.fromarray(frame, mode="RGB") for frame in video_np]
    return frames


__all__ = ["pil_frames_to_uint8_tensor", "uint8_tensor_to_pil_frames"]
