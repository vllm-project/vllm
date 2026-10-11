# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark early video frame resizing vs baseline decoding.

Measures memory consumption, decode time, and verifies bit-identity
with the Hugging Face Qwen3-VL video processor.
Can run on synthetic video if no input video is supplied.

Usage:
    python benchmarks/video_early_resize_benchmark.py [video.mp4 ...]
"""

import argparse
import contextlib
import os
import statistics
import tempfile
import time

import cv2
import numpy as np
import torch
from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
    Qwen3VLVideoProcessor,
)
from transformers.video_utils import VideoMetadata

from vllm.multimodal.media import ImageMediaIO, VideoMediaIO


def create_synthetic_video(
    path: str, num_frames: int = 60, height: int = 720, width: int = 1280
):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, 24.0, (width, height))
    for f in range(num_frames):
        frame = np.full((height, width, 3), (f * 5) % 256, dtype=np.uint8)
        out.write(frame)
    out.release()


def main():
    parser = argparse.ArgumentParser(description="Benchmark early video resizing")
    parser.add_argument("videos", nargs="*", help="Optional path(s) to video files")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repetitions")
    args = parser.parse_args()

    torch.set_num_threads(1)
    vp = Qwen3VLVideoProcessor()
    cfg = {
        "shortest_edge": vp.size["shortest_edge"],
        "longest_edge": vp.size["longest_edge"],
        "factor": vp.patch_size * vp.merge_size,
        "temporal_factor": vp.temporal_patch_size,
    }

    video_paths = list(args.videos)
    temp_files = []
    if not video_paths:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_f:
            tmp = tmp_f
        create_synthetic_video(tmp.name, num_frames=60, height=720, width=1280)
        video_paths.append(tmp.name)
        temp_files.append(tmp.name)
        print(f"Generated synthetic 720p video at {tmp.name}")

    try:
        for path in video_paths:
            with open(path, "rb") as f:
                data = f.read()

            io_base = VideoMediaIO(
                ImageMediaIO(), num_frames=-1, fps=2, video_backend="qwen3_vl"
            )
            io_early = VideoMediaIO(
                ImageMediaIO(),
                num_frames=-1,
                fps=2,
                video_backend="qwen3_vl",
                _resize_size_config=cfg,
            )

            print(f"\nEvaluating: {path} ({len(data) / 1024:.1f} KB)")

            # Baseline decode
            base_times = []
            base_frames = None
            base_meta = None
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                base_frames, base_meta = io_base.load_bytes(data).media
                base_times.append(time.perf_counter() - t0)

            # Early resize decode
            early_times = []
            early_frames = None
            early_meta = None
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                early_frames, early_meta = io_early.load_bytes(data).media
                early_times.append(time.perf_counter() - t0)

            base_mem = base_frames.nbytes / (1024 * 1024)
            early_mem = early_frames.nbytes / (1024 * 1024)
            base_dec_t = statistics.median(base_times)
            early_dec_t = statistics.median(early_times)

            print(
                f"  Baseline decode:     {base_dec_t:.4f}s | "
                f"Frame memory: {base_mem:.2f} MiB {base_frames.shape}"
            )
            print(
                f"  Early resize decode: {early_dec_t:.4f}s | "
                f"Frame memory: {early_mem:.2f} MiB {early_frames.shape}"
            )
            print(
                f"  Memory reduction:    "
                f"{((base_mem - early_mem) / base_mem * 100):.1f}%"
            )

            # Check processor output bit-identity
            md = VideoMetadata(
                total_num_frames=base_meta["total_num_frames"],
                fps=base_meta["fps"],
                duration=base_meta["duration"],
            )
            out_base = vp(
                videos=[base_frames],
                video_metadata=[md],
                do_sample_frames=False,
                return_tensors="pt",
            )
            out_early = vp(
                videos=[early_frames],
                video_metadata=[md],
                do_sample_frames=False,
                return_tensors="pt",
            )

            same_grid = torch.equal(
                out_base["video_grid_thw"], out_early["video_grid_thw"]
            )
            same_pixels = torch.equal(
                out_base["pixel_values_videos"], out_early["pixel_values_videos"]
            )
            print(f"  Bit-identical grid:   {same_grid}")
            print(f"  Bit-identical pixels: {same_pixels}")

    finally:
        for tmp in temp_files:
            with contextlib.suppress(OSError):
                os.remove(tmp)


if __name__ == "__main__":
    main()
